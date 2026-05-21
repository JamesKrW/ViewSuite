# render_stress_test.py  (realtime JSONL progress + JSON-safe summary)
from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from view_suite.envs.scannet_proxy_task.interactive_view_planning import InteractiveViewPlanning

ACTION_TEMPLATE = "<think>stress</think><action>select_view({view})|</action>"


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Compute linear-interpolated percentile on a sorted list."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pct = max(0.0, min(100.0, pct))
    idx = (len(values) - 1) * pct / 100.0
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return values[int(idx)]
    frac = idx - lower
    return values[lower] * (1.0 - frac) + values[upper] * frac


@dataclass
class WorkerConfig:
    """Configuration for a single logical worker."""
    worker_id: int
    env_config: Dict[str, Any]
    resets: int
    steps_per_reset: int
    view_name: str
    sleep_ms: float
    seed: Optional[int] = None
    scene_id: Optional[str] = None
    # Realtime progress reporting
    progress_q: Optional[asyncio.Queue] = None  # queue of JSON-serializable events
    progress_every_step: bool = True            # emit a JSONL line for each step


@dataclass
class WorkerResult:
    """Aggregated result for a worker."""
    worker_id: int
    resets: int = 0
    steps: int = 0
    successes: int = 0
    failures: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scene_id: Optional[str] = None
    seeds: List[int] = field(default_factory=list)

    def extend(self, latency: float, ok: bool, error: Optional[str]) -> None:
        """Append a step result."""
        self.latencies.append(latency)
        self.steps += 1
        if ok:
            self.successes += 1
        else:
            self.failures += 1
            if error:
                self.errors.append(error)


def _worker_result_to_json(r: WorkerResult) -> Dict[str, Any]:
    """Convert WorkerResult into a JSON-serializable dict."""
    # You can also use `asdict(r)`, but being explicit is safer and clearer.
    return {
        "worker_id": r.worker_id,
        "scene_id": r.scene_id,
        "resets": r.resets,
        "steps": r.steps,
        "successes": r.successes,
        "failures": r.failures,
        "latencies": list(r.latencies),
        "errors": list(r.errors),
        "seeds": list(r.seeds),
    }


async def _progress_jsonl_writer(
    q: asyncio.Queue,
    jsonl_path: str,
    *,
    fsync: bool = False,
) -> None:
    """
    Background task that appends JSON events to a .jsonl file in realtime.
    Each event must be JSON-serializable; a None item signals termination.
    """
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Line-buffered text mode; flush after each write
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        while True:
            item = await q.get()
            if item is None:
                # Sentinel to stop the writer
                break
            try:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                if fsync:
                    os.fsync(f.fileno())
            except Exception as exc:  # best-effort logging into the same file
                # Write an error event to help diagnose file issues.
                f.write(json.dumps({
                    "type": "writer_error",
                    "ts": time.time(),
                    "error": repr(exc),
                }, ensure_ascii=False) + "\n")
                f.flush()


async def _run_worker(cfg: WorkerConfig) -> WorkerResult:
    """
    Execute resets and steps for a worker.
    Emits progress events to cfg.progress_q if provided.
    """
    result = WorkerResult(worker_id=cfg.worker_id, scene_id=cfg.scene_id)
    env = InteractiveViewPlanning(cfg.env_config)
    action = ACTION_TEMPLATE.format(view=cfg.view_name)

    # Emit worker_start
    if cfg.progress_q is not None:
        await cfg.progress_q.put({
            "type": "worker_start",
            "ts": time.time(),
            "worker_id": cfg.worker_id,
            "scene_id": cfg.scene_id,
            "resets": cfg.resets,
            "steps_per_reset": cfg.steps_per_reset,
            "view_name": cfg.view_name,
        })

    try:
        for reset_idx in range(cfg.resets):
            if cfg.seed is None:
                raise ValueError(f"worker {cfg.worker_id} missing required seed")
            seed = cfg.seed

            # Emit reset_start
            if cfg.progress_q is not None:
                await cfg.progress_q.put({
                    "type": "reset_start",
                    "ts": time.time(),
                    "worker_id": cfg.worker_id,
                    "scene_id": cfg.scene_id,
                    "reset": reset_idx,
                    "seed": seed,
                })

            obs, info = await env.reset(seed=seed)
            result.resets += 1
            result.seeds.append(seed)

            available = set(info.get("named_views") or ())
            if cfg.view_name not in available:
                # Emit missing_view failure
                err = (
                    f"worker={cfg.worker_id} scene={cfg.scene_id} seed={seed} "
                    f"reset={reset_idx} missing view '{cfg.view_name}' (available={sorted(available)})"
                )
                if cfg.progress_q is not None:
                    await cfg.progress_q.put({
                        "type": "missing_view",
                        "ts": time.time(),
                        "worker_id": cfg.worker_id,
                        "scene_id": cfg.scene_id,
                        "reset": reset_idx,
                        "seed": seed,
                        "view_name": cfg.view_name,
                        "available": sorted(available),
                        "fail_steps": cfg.steps_per_reset,
                        "error": err,
                    })
                result.failures += cfg.steps_per_reset
                result.errors.append(err)
                continue

            for step_idx in range(cfg.steps_per_reset):
                started = time.perf_counter()
                error_msg = None
                ok = True
                try:
                    step_obs, _, _, step_info = await env.step(action)
                except Exception as exc:  # noqa: BLE001
                    latency = time.perf_counter() - started
                    err = (
                        f"worker={cfg.worker_id} scene={cfg.scene_id} seed={seed} "
                        f"reset={reset_idx} step={step_idx} exception={exc}"
                    )
                    ok = False
                    error_msg = err
                    result.extend(latency, False, err)

                    # Emit step_end with exception
                    if cfg.progress_q is not None and cfg.progress_every_step:
                        await cfg.progress_q.put({
                            "type": "step_end",
                            "ts": time.time(),
                            "worker_id": cfg.worker_id,
                            "scene_id": cfg.scene_id,
                            "reset": reset_idx,
                            "step": step_idx,
                            "seed": seed,
                            "latency_s": latency,
                            "ok": False,
                            "error": err,
                        })
                    continue

                latency = time.perf_counter() - started
                if isinstance(step_info, dict):
                    error_msg = step_info.get("error")
                if not error_msg and isinstance(step_obs, dict):
                    raw = step_obs.get("obs_str") or ""
                    if isinstance(raw, str) and raw.lower().startswith("format: error"):
                        error_msg = raw
                ok = error_msg is None
                if error_msg:
                    error_msg = (
                        f"worker={cfg.worker_id} scene={cfg.scene_id} seed={seed} "
                        f"reset={reset_idx} step={step_idx} error={error_msg}"
                    )
                result.extend(latency, ok, error_msg)

                # Emit step_end on every step (optional)
                if cfg.progress_q is not None and cfg.progress_every_step:
                    await cfg.progress_q.put({
                        "type": "step_end",
                        "ts": time.time(),
                        "worker_id": cfg.worker_id,
                        "scene_id": cfg.scene_id,
                        "reset": reset_idx,
                        "step": step_idx,
                        "seed": seed,
                        "latency_s": latency,
                        "ok": ok,
                        "error": error_msg,
                    })

                if cfg.sleep_ms > 0:
                    await asyncio.sleep(cfg.sleep_ms / 1000.0)

    finally:
        await env.close()
        # Emit worker_end
        if cfg.progress_q is not None:
            await cfg.progress_q.put({
                "type": "worker_end",
                "ts": time.time(),
                "worker_id": cfg.worker_id,
                "scene_id": cfg.scene_id,
                "resets": result.resets,
                "steps": result.steps,
                "successes": result.successes,
                "failures": result.failures,
                "error_count": len(result.errors),
            })

    return result


def _summarize(results: Iterable[WorkerResult]) -> Dict[str, Any]:
    """Aggregate summary stats across workers."""
    aggregated: Dict[str, Any] = {}
    total_resets = sum(r.resets for r in results)
    total_steps = sum(r.steps for r in results)
    total_success = sum(r.successes for r in results)
    total_fail = sum(r.failures for r in results)
    latencies = sorted(lat for r in results for lat in r.latencies)
    total_latency = sum(latencies)

    aggregated["total_resets"] = total_resets
    aggregated["total_steps"] = total_steps
    aggregated["successes"] = total_success
    aggregated["failures"] = total_fail
    aggregated["success_rate"] = (total_success / total_steps) if total_steps else 0.0
    aggregated["total_latency_s"] = total_latency

    if latencies:
        aggregated["latency_mean_s"] = statistics.fmean(latencies)
        aggregated["latency_median_s"] = statistics.median(latencies)
        aggregated["latency_p95_s"] = _percentile(latencies, 95.0)
        aggregated["latency_p99_s"] = _percentile(latencies, 99.0)
    else:
        aggregated["latency_mean_s"] = None
        aggregated["latency_median_s"] = None
        aggregated["latency_p95_s"] = None
        aggregated["latency_p99_s"] = None

    aggregated["errors"] = [err for r in results for err in r.errors]
    aggregated["scene_assignments"] = {
        r.worker_id: {"scene_id": r.scene_id, "seeds": list(r.seeds)} for r in results
    }
    aggregated["error_details"] = [
        {
            "worker_id": r.worker_id,
            "scene_id": r.scene_id,
            "seeds": list(r.seeds),
            "messages": list(r.errors),
        }
        for r in results
        if r.errors
    ]
    return aggregated


def _collect_scene_row_ids(jsonl_path: str) -> Dict[str, int]:
    """Scan a dataset JSONL and map first occurrence of each scene_id to its line index."""
    scene_to_row: Dict[str, int] = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scene_id = payload.get("scene_id")
                if isinstance(scene_id, str) and scene_id not in scene_to_row:
                    scene_to_row[scene_id] = row_idx
    except FileNotFoundError:
        pass
    return scene_to_row


def _write_status(status_path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    """Persist worker status snapshot to disk (JSON, not JSONL)."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"workers": records}
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


async def _run_stress(
    client_url: str,
    jsonl_path: str,
    *,
    client_origin: Optional[str] = None,
    client_open_timeout: Optional[float] = None,
    client_max_inflight: Optional[int] = None,
    dataset_root: Optional[str] = None,
    steps_per_reset: int = 4,
    view_name: str = "init_view",
    total_envs: int = 0,
    max_concurrent_env: Optional[int] = None,
    sleep_ms: float = 0.0,
    render_width: int = 300,
    render_height: int = 300,
    status_path: Optional[str] = None,
    # Realtime JSONL options
    progress_jsonl_path: Optional[str] = None,
    progress_fsync: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate stress test across N logical workers with M concurrent.
    Emits realtime JSONL events if progress_jsonl_path is provided.
    """
    env_base: Dict[str, Any] = {
        "jsonl_path": jsonl_path,
        "dataset_root": dataset_root,
        "render_backend": "client",
        "client_url": client_url,
        "client_origin": client_origin,
        "client_open_timeout": client_open_timeout,
        "client_max_inflight": client_max_inflight,
        "max_turns": steps_per_reset + 1,
        "render_width": render_width,
        "render_height": render_height,
    }

    scene_row_ids = _collect_scene_row_ids(jsonl_path)
    if not scene_row_ids:
        raise ValueError(f"No scene ids found in {jsonl_path}")

    scenes: List[Tuple[str, int]] = sorted(scene_row_ids.items())
    effective_total = total_envs if total_envs > 0 else len(scenes)
    if effective_total <= 0:
        raise ValueError("total_envs must be > 0 or the dataset must contain scenes")

    # Create progress queue + writer task if JSONL requested
    progress_q: Optional[asyncio.Queue] = None
    writer_task: Optional[asyncio.Task] = None
    if progress_jsonl_path:
        progress_q = asyncio.Queue(maxsize=1024)
        writer_task = asyncio.create_task(
            _progress_jsonl_writer(progress_q, progress_jsonl_path, fsync=progress_fsync)
        )
        # Emit a run_start marker
        await progress_q.put({
            "type": "run_start",
            "ts": time.time(),
            "client_url": client_url,
            "jsonl_path": jsonl_path,
            "total_envs": effective_total,
            "max_concurrent_env": max_concurrent_env or effective_total,
            "steps_per_reset": steps_per_reset,
            "render_size": [render_width, render_height],
        })

    workers: List[WorkerConfig] = []
    for idx in range(effective_total):
        scene_id, row_id = scenes[idx % len(scenes)]
        workers.append(
            WorkerConfig(
                worker_id=idx,
                env_config=dict(env_base),
                resets=1,
                steps_per_reset=steps_per_reset,
                view_name=view_name,
                sleep_ms=sleep_ms,
                seed=row_id,
                scene_id=scene_id,
                progress_q=progress_q,
                progress_every_step=True,
            )
        )

    max_concurrent = max_concurrent_env or effective_total
    max_concurrent = max(1, min(max_concurrent, len(workers)))
    results: List[WorkerResult] = []
    status_records: Dict[str, Dict[str, Any]] = {}
    status_path_obj = Path(status_path) if status_path else None
    summary_cache: Optional[Dict[str, Any]] = None

    worker_iter = iter(workers)
    active_tasks: List[asyncio.Task[WorkerResult]] = []
    task_to_cfg: Dict[asyncio.Task[WorkerResult], WorkerConfig] = {}
    start_time = time.perf_counter()

    def _schedule_next() -> None:
        """Launch the next worker if any remain."""
        try:
            cfg = next(worker_iter)
        except StopIteration:
            return
        task = asyncio.create_task(_run_worker(cfg))
        active_tasks.append(task)
        task_to_cfg[task] = cfg

    for _ in range(min(max_concurrent, len(workers))):
        _schedule_next()

    while active_tasks:
        done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        active_tasks = list(pending)

        for task in done:
            cfg = task_to_cfg.pop(task, None)
            try:
                result = task.result()
            except Exception as exc:  # noqa: BLE001
                if cfg is not None:
                    result = WorkerResult(worker_id=cfg.worker_id, scene_id=cfg.scene_id)
                    result.errors.append(
                        f"worker={cfg.worker_id} scene={cfg.scene_id} seed={cfg.seed} exception={exc}"
                    )
                    result.failures = cfg.steps_per_reset * cfg.resets
                else:
                    raise
            results.append(result)

            # Emit a worker_summary row to JSONL as well
            if progress_q is not None:
                await progress_q.put({
                    "type": "worker_summary",
                    "ts": time.time(),
                    "worker_id": result.worker_id,
                    "scene_id": result.scene_id,
                    "resets": result.resets,
                    "steps": result.steps,
                    "successes": result.successes,
                    "failures": result.failures,
                    "error_count": len(result.errors),
                })

            # Keep writing the JSON snapshot (old behavior)
            if status_path_obj is not None:
                status_records[str(result.worker_id)] = {
                    "scene_id": result.scene_id,
                    "seeds": list(result.seeds),
                    "status": "success" if result.failures == 0 else "failed",
                    "successes": result.successes,
                    "failures": result.failures,
                    "errors": list(result.errors),
                }
                summary_cache = _summarize(results)
                elapsed = time.perf_counter() - start_time
                steps_done = sum(r.steps for r in results)
                summary_cache["wall_time_s"] = elapsed
                summary_cache["completed_workers"] = len(results)
                summary_cache["steps_completed"] = steps_done
                summary_cache["avg_wall_per_worker_s"] = elapsed / len(results) if results else None
                summary_cache["avg_wall_per_step_s"] = elapsed / steps_done if steps_done else None
                payload = {
                    "workers": status_records,
                    "summary": summary_cache,  # keep full summary in incremental writes
                }
                status_path_obj.parent.mkdir(parents=True, exist_ok=True)
                status_path_obj.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            if len(active_tasks) < max_concurrent:
                _schedule_next()

    # Final summary (also write to status_path if provided)
    summary = summary_cache if summary_cache is not None else _summarize(results)
    elapsed = time.perf_counter() - start_time
    steps_done = sum(r.steps for r in results)
    summary["wall_time_s"] = elapsed
    summary["completed_workers"] = len(results)
    summary["steps_completed"] = steps_done
    summary["avg_wall_per_worker_s"] = elapsed / len(results) if results else None
    summary["avg_wall_per_step_s"] = elapsed / steps_done if steps_done else None

    # Convert WorkerResult objects to plain dicts for JSON
    summary["per_worker"] = [_worker_result_to_json(r) for r in results]
    # (Alternatively: summary["per_worker"] = [asdict(r) for r in results])

    if status_path_obj is not None:
        final_payload = {
            "workers": status_records,
            "summary": summary,  # include full summary at the end
        }
        status_path_obj.parent.mkdir(parents=True, exist_ok=True)
        status_path_obj.write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Emit run_end + stop writer
    if progress_q is not None:
        await progress_q.put({
            "type": "run_end",
            "ts": time.time(),
            "completed_workers": len(results),
            "total_steps": steps_done,
            "wall_time_s": elapsed,
        })
        await progress_q.put(None)  # sentinel
        if writer_task is not None:
            await writer_task

    return summary


def main(
    client_url: str,
    jsonl_path: str,
    *,
    client_origin: Optional[str] = None,
    client_open_timeout: Optional[float] = None,
    client_max_inflight: Optional[int] = None,
    dataset_root: Optional[str] = None,
    steps_per_reset: int = 6,
    view_name: str = "init_view",
    total_envs: int = 128,
    max_concurrent_env: Optional[int] = 32,
    sleep_ms: float = 0.0,
    render_width: int = 300,
    render_height: int = 300,
    status_path: Optional[str] = "./render_stress_status.json",
    # NEW CLI args
    progress_jsonl_path: Optional[str] = "./render_stress_progress.jsonl",
    progress_fsync: bool = False,
) -> str:
    """
    Stress-test the ScanNet render client/server by firing select_view requests across scenes.
    A realtime JSONL log is written if progress_jsonl_path is provided.
    """
    summary = asyncio.run(
        _run_stress(
            client_url=client_url,
            jsonl_path=jsonl_path,
            client_origin=client_origin,
            client_open_timeout=client_open_timeout,
            client_max_inflight=client_max_inflight,
            dataset_root=dataset_root,
            steps_per_reset=steps_per_reset,
            view_name=view_name,
            total_envs=total_envs,
            max_concurrent_env=max_concurrent_env,
            sleep_ms=sleep_ms,
            render_width=render_width,
            render_height=render_height,
            status_path=status_path,
            progress_jsonl_path=progress_jsonl_path,
            progress_fsync=progress_fsync,
        )
    )
    # Pretty-print JSON so Fire prints readable output (final summary)
    return json.dumps(summary, indent=2, sort_keys=True)


if __name__ == "__main__":
    import fire
    fire.Fire(main)

# Example:
# python ViewSuite/view_suite/scannet/tests/render_stress_test.py \
#   --client_url=ws://0.0.0.0:8766/render?token=my-secret-render-token-123 \
#   --jsonl_path=data/scannet_proxy_task_test/interactive_view_planning_qa.jsonl \
#   --total_envs=360 --max_concurrent_env=120 --steps_per_reset=6 \
#   --progress_jsonl_path=outputs/render_stress_progress.jsonl
