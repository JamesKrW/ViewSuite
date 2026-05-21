"""
Periodic helper that monitors checkpoint directories during training.

Detects when the final checkpoint is ready, copies it to the designated
output path, and cleans up intermediate checkpoints to save disk space.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from graphrl.utils.periodic_task import PeriodicTask

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ["config.json"]
_WEIGHT_PATTERNS = ["*.safetensors", "*.bin"]
_STABILITY_DELAY = 30
_STABILITY_ROUNDS = 2


class CheckpointMonitor(PeriodicTask):
    """
    Monitors a checkpoint source directory and moves the final checkpoint
    to a designated output path when ready.

    For VAGEN: detects ``global_step_X/actor/huggingface/`` where X >= target_steps.
    For LLaMA-Factory: detects ``config.json`` in the output dir.

    Args:
        source_dir: Directory where the training backend writes checkpoints.
        dest_path: Directory where the final model should be placed.
        target_steps: Expected final global_step (VAGEN). None for LLaMA-Factory.
        interval: Seconds between monitoring ticks.
        clean_intermediates: Whether to delete intermediate checkpoints.
    """

    def __init__(
        self,
        source_dir: Path,
        dest_path: Path,
        target_steps: Optional[int],
        interval: float,
        clean_intermediates: bool = True,
    ):
        super().__init__(interval, name="CheckpointMonitor")
        self.source_dir = Path(source_dir)
        self.dest_path = Path(dest_path)
        self.target_steps = target_steps
        self.clean_intermediates = clean_intermediates
        self._final_detected = False

    def tick(self) -> None:
        if self._final_detected:
            return
        if not self.source_dir.exists():
            return

        if self.target_steps is not None:
            self._check_vagen_checkpoints()
        else:
            self._check_llama_factory_checkpoints()

    def _check_vagen_checkpoints(self) -> None:
        ckpts = self._list_vagen_checkpoints()
        if not ckpts:
            return

        latest_dir, latest_step = ckpts[-1]
        if latest_step < self.target_steps:
            return

        hf_path = latest_dir / "actor" / "huggingface"
        if not hf_path.exists():
            return
        if not self._is_checkpoint_complete(hf_path):
            return
        if not self._is_checkpoint_stable(hf_path):
            return

        self._copy_checkpoint(hf_path, self.dest_path)
        self._final_detected = True
        logger.info(
            f"[CheckpointMonitor] Final VAGEN checkpoint at step {latest_step} "
            f"copied to {self.dest_path}"
        )

        if self.clean_intermediates:
            for ckpt_dir, step in ckpts:
                if ckpt_dir != latest_dir:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
            for subdir in ["critic", "optimizer"]:
                p = latest_dir / subdir
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
            data_pt = latest_dir / "actor" / "data.pt"
            if data_pt.exists():
                data_pt.unlink()

    def _list_vagen_checkpoints(self):
        result = []
        for d in self.source_dir.iterdir():
            if d.is_dir() and d.name.startswith("global_step_"):
                try:
                    step = int(d.name.split("_")[-1])
                    result.append((d, step))
                except ValueError:
                    continue
        result.sort(key=lambda x: x[1])
        return result

    def _check_llama_factory_checkpoints(self) -> None:
        if not self._is_checkpoint_complete(self.source_dir):
            return
        if not self._is_checkpoint_stable(self.source_dir):
            return

        self._final_detected = True
        logger.info(f"[CheckpointMonitor] LLaMA-Factory training complete at {self.source_dir}")

        if self.clean_intermediates:
            for d in sorted(self.source_dir.iterdir()):
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _is_checkpoint_complete(path: Path) -> bool:
        for required in _REQUIRED_FILES:
            if not (path / required).exists():
                return False
        return any(list(path.glob(p)) for p in _WEIGHT_PATTERNS)

    @staticmethod
    def _snapshot_dir(path: Path) -> dict:
        """Take a snapshot of all files in a directory: {relative_name: size}."""
        snapshot = {}
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    snapshot[str(item.relative_to(path))] = item.stat().st_size
                except OSError:
                    pass
        return snapshot

    @staticmethod
    def _is_checkpoint_stable(path: Path) -> bool:
        """Check that the directory contents have stopped changing.

        Takes multiple snapshots of ALL files (not just weights) separated by
        ``_STABILITY_DELAY`` seconds.  Only returns True when two consecutive
        snapshots are identical — meaning no new files appeared and no existing
        file changed size.
        """
        prev = CheckpointMonitor._snapshot_dir(path)
        if not prev:
            return False
        for _ in range(_STABILITY_ROUNDS):
            time.sleep(_STABILITY_DELAY)
            curr = CheckpointMonitor._snapshot_dir(path)
            if curr != prev:
                return False
            prev = curr
        return True

    @staticmethod
    def _copy_checkpoint(src: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dest / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
