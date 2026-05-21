# actor.py - Ray Actor for ScanNet rendering
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from view_suite.scannet.render.mesh_render import MeshRenderer
from view_suite.scannet.utils.path_utils import resolve_scene_ply
from view_suite.service_http.multipart import numpy_to_png_bytes

LOGGER = logging.getLogger(__name__)


class ScanNetRenderActorImpl:
    """
    Ray Actor implementation (kept as a plain class; wrapped by ray.remote in handler).

    - Holds at most ONE loaded scene at a time (scene cache per actor).
    - If a new scene_id arrives, it replaces the old one (best-effort release).
    - Render returns PNG bytes for each task.
    """

    def __init__(
        self,
        *,
        scannet_root: str,
        forced_render_size: Optional[Tuple[int, int]] = None,
        log_level: int = logging.INFO,
    ) -> None:
        LOGGER.setLevel(log_level)
        self.scannet_root = scannet_root
        self.forced_render_size = forced_render_size

        self._lock = threading.Lock()
        self._active_scene_id: Optional[str] = None
        self._active_renderer: Optional[MeshRenderer] = None

        # Ray will set CUDA_VISIBLE_DEVICES per-actor based on num_gpus assignment.
        # So no manual binding is required here.

    def _release_renderer_locked(self) -> None:
        """Best-effort release of GPU/GL resources."""
        if self._active_renderer is None:
            return
        off = getattr(self._active_renderer, "renderer", None)
        if off is not None and hasattr(off, "release"):
            with contextlib.suppress(Exception):
                off.release()
        self._active_renderer = None
        self._active_scene_id = None

    def _ensure_scene_loaded_locked(self, scene_id: str) -> MeshRenderer:
        """Load or reuse renderer for this actor."""
        import time

        if self._active_scene_id == scene_id and self._active_renderer is not None:
            LOGGER.info("[ScanNetRender/RayActor] Scene cache HIT: scene=%s", scene_id)
            return self._active_renderer

        # replace old scene
        old_scene = self._active_scene_id
        if old_scene:
            LOGGER.info(
                "[ScanNetRender/RayActor] Scene cache MISS: evicting scene=%s, loading scene=%s",
                old_scene, scene_id
            )
        else:
            LOGGER.info("[ScanNetRender/RayActor] Scene cache EMPTY: loading scene=%s", scene_id)

        self._release_renderer_locked()

        load_start = time.time()
        ply_path = resolve_scene_ply(self.scannet_root, scene_id)
        LOGGER.info("[ScanNetRender/RayActor] Loading mesh from: %s", ply_path)
        renderer = MeshRenderer(ply_path)
        load_time = time.time() - load_start

        self._active_renderer = renderer
        self._active_scene_id = scene_id
        LOGGER.info("[ScanNetRender/RayActor] Loaded scene=%s in %.3fs", scene_id, load_time)
        return renderer

    def get_current_scene(self) -> Optional[str]:
        return self._active_scene_id

    def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[bytes]:
        """
        Render tasks for scene_id.

        tasks format is identical to your original handler:
          - mode == "cam_param": intrinsics, extrinsics, size
        """
        import time

        start_time = time.time()

        with self._lock:
            renderer = self._ensure_scene_loaded_locked(scene_id)
            forced = self.forced_render_size

        load_time = time.time() - start_time

        LOGGER.info(
            "[ScanNetRender/RayActor] Rendering scene=%s tasks=%d (load_time=%.3fs)",
            scene_id, len(tasks), load_time
        )

        out: List[bytes] = []
        task_times = []

        for idx, task in enumerate(tasks):
            task_start = time.time()

            # Determine render size
            if forced is not None:
                w, h = forced
            else:
                size = task.get("size") or [300, 300]
                w, h = int(size[0]), int(size[1])

            mode = (task.get("mode") or "").lower()

            if mode == "cam_param":
                K = np.array(task["intrinsics"], dtype=float)
                T = np.array(task["extrinsics"], dtype=float)
                img_array = renderer.render_image_from_cam_param(K, T, width=w, height=h)
                out.append(numpy_to_png_bytes(img_array.astype(np.uint8)))
            else:
                LOGGER.warning(
                    "[ScanNetRender/RayActor] Unknown mode='%s' task=%d -> transparent",
                    mode, idx
                )
                out.append(numpy_to_png_bytes(np.zeros((h, w, 4), dtype=np.uint8)))

            task_time = time.time() - task_start
            task_times.append(task_time)

        total_time = time.time() - start_time
        avg_task_time = sum(task_times) / len(task_times) if task_times else 0

        LOGGER.info(
            "[ScanNetRender/RayActor] Completed scene=%s tasks=%d total_time=%.3fs avg_per_task=%.3fs",
            scene_id, len(tasks), total_time, avg_task_time
        )

        return out
