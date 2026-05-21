# handler.py - ScanNet Render Handler (Ray-based) for HTTP Service
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from view_suite.service_http.handler import BaseHandler, HandlerResult

# Ray actor impl
from .actor import ScanNetRenderActorImpl

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)


def _normalize_gpu_ids(value: Sequence[int] | str | int | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        tokens = [str(value)]
    elif isinstance(value, str):
        tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    else:
        tokens = [str(v).strip() for v in value]
    out: list[int] = []
    for tok in tokens:
        try:
            out.append(int(tok))
        except Exception:
            LOGGER.warning("Ignoring invalid GPU id token: %s", tok)
    return out


def _normalize_forced_size(value: Sequence[int] | int | str | None) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        v = max(1, int(value))
        return (v, v)
    if isinstance(value, str):
        tokens = value.replace("x", ",").replace("X", ",").split(",")
        vals = [int(tok.strip()) for tok in tokens if tok.strip()]
    else:
        vals = [int(x) for x in value]
    if len(vals) != 2:
        raise ValueError("forced_render_size must be exactly two ints (width,height)")
    return (max(1, vals[0]), max(1, vals[1]))


def _infer_visible_gpu_count(normalized_gpu_ids: list[int]) -> int:
    """
    Prefer explicit gpu_ids length; otherwise try env CUDA_VISIBLE_DEVICES; otherwise 0.
    """
    if normalized_gpu_ids:
        return len(normalized_gpu_ids)
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 0
    toks = [t.strip() for t in cvd.split(",") if t.strip()]
    # CUDA_VISIBLE_DEVICES can be something like "0,1,2"
    # If it is "0" that's 1.
    return len(toks)


@dataclass
class _ActorSlot:
    """
    Represents one Ray actor.
    """
    actor: Any  # ray actor handle
    current_scene: Optional[str] = None
    last_used: float = 0.0
    inflight: int = 0


class _RayActorPool:
    """
    Fixed-size actor pool:
      - sticky dispatch by scene_id if possible
      - prefer idle empty slots
      - LRU eviction when all slots have scenes and we need a new one
      - simple crash recovery: recreate actor and retry once
    """

    def __init__(
        self,
        *,
        max_actors: int,
        scannet_root: str,
        gpu_ids: Sequence[int] | str | None,
        forced_render_size: Optional[Tuple[int, int]],
        log_level: int,
        crash_cooldown_s: float = 0.5,
    ) -> None:
        self.max_actors = max(1, int(max_actors))
        self.scannet_root = scannet_root
        self.forced_render_size = forced_render_size
        self.log_level = log_level
        self.crash_cooldown_s = float(crash_cooldown_s)

        self._metrics: Counter[str] = Counter()
        self._lock = asyncio.Lock()

        # ---- Ray init (best-effort) ----
        import ray

        normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)

        # If user explicitly passed gpu_ids, constrain CUDA_VISIBLE_DEVICES for this process;
        # Ray workers inherit it, so scheduling will be limited to that set.
        if normalized_gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in normalized_gpu_ids)

        # Support connecting to cluster if RAY_ADDRESS is set.
        ray_address = os.getenv("RAY_ADDRESS")
        if not ray.is_initialized():
            ray.init(
                address=ray_address or None,
                ignore_reinit_error=True,
                log_to_driver=True,  # Enable actor logs to main process
                logging_level=self.log_level,  # Set Ray's logging level
            )

        # Compute per-actor GPU fraction:
        # - if visible_gpus == 0 => 0
        # - if max_actors <= visible_gpus => 1 GPU per actor
        # - else => share GPUs fractionally: visible_gpus / max_actors
        visible_gpus = _infer_visible_gpu_count(normalized_gpu_ids)
        if visible_gpus <= 0:
            num_gpus_per_actor = 0
        elif visible_gpus == 1:
            num_gpus_per_actor = 0 # no specified gpu num
        elif self.max_actors <= visible_gpus:
            num_gpus_per_actor = 1.0
        else:
            num_gpus_per_actor = float(visible_gpus) / float(self.max_actors)

        self._num_gpus_per_actor = num_gpus_per_actor

        # Wrap the impl with ray.remote dynamically (so we can parameterize resources).
        RemoteActor = ray.remote(num_cpus=1, num_gpus=num_gpus_per_actor)(ScanNetRenderActorImpl) # default in single gpu mode

        self._RemoteActor = RemoteActor
        self._slots: List[_ActorSlot] = []
        for _ in range(self.max_actors):
            a = self._create_actor()
            self._slots.append(_ActorSlot(actor=a))

        self._scene_to_idx: Dict[str, int] = {}

    def _create_actor(self):
        return self._RemoteActor.remote(
            scannet_root=self.scannet_root,
            forced_render_size=self.forced_render_size,
            log_level=self.log_level,
        )

    def _pick_slot_locked(self, scene_id: str) -> int:
        """
        Must hold self._lock.
        Strategy:
          1) sticky if scene_id already mapped
          2) find idle+empty (current_scene is None and inflight==0)
          3) find idle slot (inflight==0), prefer empty else LRU
          4) if all busy, LRU among all (will queue on actor)
        """
        mapped = self._scene_to_idx.get(scene_id)
        if mapped is not None:
            return mapped

        # idle + empty
        for i, s in enumerate(self._slots):
            if s.current_scene is None and s.inflight == 0:
                s.current_scene = scene_id
                self._scene_to_idx[scene_id] = i
                return i

        # any idle
        idle = [i for i, s in enumerate(self._slots) if s.inflight == 0]
        if idle:
            # prefer empty
            for i in idle:
                if self._slots[i].current_scene is None:
                    self._slots[i].current_scene = scene_id
                    self._scene_to_idx[scene_id] = i
                    return i

            # LRU among idle
            i = min(idle, key=lambda k: self._slots[k].last_used)
            prev = self._slots[i].current_scene
            if prev:
                self._scene_to_idx.pop(prev, None)
            self._slots[i].current_scene = scene_id
            self._scene_to_idx[scene_id] = i
            return i

        # all busy -> global LRU
        i = min(range(len(self._slots)), key=lambda k: self._slots[k].last_used)
        prev = self._slots[i].current_scene
        if prev:
            self._scene_to_idx.pop(prev, None)
        self._slots[i].current_scene = scene_id
        self._scene_to_idx[scene_id] = i
        return i

    async def _ray_get_async(self, obj_ref):
        """
        Safe awaitable ray.get via thread executor (works regardless of Ray asyncio mode).
        """
        import ray
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, ray.get, obj_ref)

    async def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        import ray
        from ray.exceptions import RayActorError

        start_time = time.monotonic()

        async with self._lock:
            idx = self._pick_slot_locked(scene_id)
            slot = self._slots[idx]
            slot.inflight += 1
            slot.last_used = time.monotonic()
            actor = slot.actor
            was_cached = (slot.current_scene == scene_id)

        LOGGER.info(
            "[ScanNetRender/RayPool] Dispatching scene=%s tasks=%d to slot=%d (cached=%s, inflight=%d)",
            scene_id, len(tasks), idx, was_cached, slot.inflight
        )

        try:
            self._metrics["submit"] += 1
            ref = actor.render.remote(scene_id, tasks)
            result = await self._ray_get_async(ref)
            elapsed = time.monotonic() - start_time
            LOGGER.info(
                "[ScanNetRender/RayPool] Completed scene=%s tasks=%d slot=%d elapsed=%.3fs",
                scene_id, len(tasks), idx, elapsed
            )
            return result
        except RayActorError:
            # Actor crashed: recreate and retry once
            self._metrics["actor_crash"] += 1
            LOGGER.error(
                "[ScanNetRender/RayPool] RayActorError on slot=%d; respawn after %.2fs",
                idx, self.crash_cooldown_s,
            )

            async with self._lock:
                # remove mapping of its current scene
                prev_scene = self._slots[idx].current_scene
                if prev_scene:
                    self._scene_to_idx.pop(prev_scene, None)

                # best-effort kill old actor
                with contextlib.suppress(Exception):
                    ray.kill(self._slots[idx].actor, no_restart=True)

                await asyncio.sleep(self.crash_cooldown_s)

                # respawn
                self._slots[idx].actor = self._create_actor()
                self._slots[idx].current_scene = scene_id
                self._scene_to_idx[scene_id] = idx
                self._slots[idx].last_used = time.monotonic()
                actor = self._slots[idx].actor

            # retry once
            try:
                self._metrics["actor_crash_retry"] += 1
                ref = actor.render.remote(scene_id, tasks)
                return await self._ray_get_async(ref)
            except RayActorError:
                self._metrics["actor_crash_repeated"] += 1
                LOGGER.error("[ScanNetRender/RayPool] repeated RayActorError slot=%d; giving up", idx)
                raise
        finally:
            async with self._lock:
                self._slots[idx].inflight = max(0, self._slots[idx].inflight - 1)
                self._slots[idx].last_used = time.monotonic()

    async def warm(self, scene_ids: Sequence[str]) -> None:
        for s in scene_ids:
            if not s:
                continue
            try:
                await self.render(s, [])
            except Exception as exc:
                LOGGER.warning("[ScanNetRender/RayPool] warm failed scene=%s err=%s", s, exc)
                self._metrics["warm_fail"] += 1

    async def aclose(self) -> None:
        import ray
        # best-effort kill actors
        for s in self._slots:
            with contextlib.suppress(Exception):
                ray.kill(s.actor, no_restart=True)

    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)


class ScanNetRenderHandler(BaseHandler):
    """
    Ray-based ScanNet render handler for HTTP service framework.

    Input meta format (unchanged):
      {
        "scene_id": "scene0011_00",
        "tasks": [ {mode, intrinsics, extrinsics, size}, ... ]
      }

    Output (unchanged):
      - meta: {"scene_id": "...", "count": N}
      - encoded_images: List[bytes] (PNG frames)
    """

    def __init__(
        self,
        max_workers: int = 4,
        scannet_root: str = "",
        log_level: int = logging.INFO,
        gpu_ids: Sequence[int] | str | None = None,
        forced_render_size: Sequence[int] | int | str | None = None,
        *,
        crash_cooldown_s: float = 0.5,
        **kwargs: Any
    ):
        LOGGER.setLevel(log_level)

        normalized_size = _normalize_forced_size(forced_render_size)

        # NOTE: Keep the constructor signature unchanged; internally we treat max_workers as max_actors.
        self.pool = _RayActorPool(
            max_actors=max_workers,
            scannet_root=scannet_root,
            gpu_ids=gpu_ids,
            forced_render_size=normalized_size,
            log_level=log_level,
            crash_cooldown_s=crash_cooldown_s,
        )

        self._metrics: Counter[str] = Counter()

    async def handle(self, meta: Dict[str, Any], images: List[Image.Image]) -> HandlerResult:
        request_start = time.monotonic()
        scene_id: str = meta.get("scene_id") or ""
        tasks: List[Dict[str, Any]] = meta.get("tasks") or []

        if not scene_id:
            LOGGER.error("[ScanNetRender] Missing scene_id in request")
            return HandlerResult(meta={"error": "Missing scene_id"}, images=[])

        if not isinstance(tasks, list):
            LOGGER.error("[ScanNetRender] Invalid tasks format")
            return HandlerResult(meta={"error": "Invalid tasks format"}, images=[])

        LOGGER.info("[ScanNetRender/Ray] Received scene_id=%s tasks=%d", scene_id, len(tasks))

        try:
            rendered_images = await self.pool.render(scene_id, tasks)
            self._metrics["success"] += 1
            self._metrics["images_returned"] += len(rendered_images)

            total_elapsed = time.monotonic() - request_start
            LOGGER.info(
                "[ScanNetRender/Ray] Request completed scene_id=%s tasks=%d images=%d total_time=%.3fs",
                scene_id, len(tasks), len(rendered_images), total_elapsed
            )

            return HandlerResult(
                meta={"scene_id": scene_id, "count": len(rendered_images)},
                encoded_images=rendered_images,
                image_format="PNG",
                image_mime="image/png",
            )
        except FileNotFoundError as exc:
            self._metrics["not_found"] += 1
            LOGGER.error("[ScanNetRender/Ray] Scene not found: %s", exc)
            return HandlerResult(meta={"error": f"Scene not found: {exc}"}, images=[])
        except Exception as exc:
            tb = traceback.format_exc()
            LOGGER.error("[ScanNetRender/Ray] Internal error: %s\n%s", exc, tb)
            self._metrics["internal_error"] += 1
            return HandlerResult(meta={"error": "Internal rendering error"}, images=[])

    async def aclose(self) -> None:
        await self.pool.aclose()

    async def warm_cache(self, scene_ids: Sequence[str]) -> None:
        await self.pool.warm(scene_ids)

    def metrics_snapshot(self) -> Dict[str, int]:
        merged = dict(self._metrics)
        try:
            merged.update({f"pool_{k}": v for k, v in self.pool.metrics().items()})
        except Exception:
            pass
        return merged
