"""
Progress detection for pipeline resume.

Scans the experiment directory for completed iteration phases and determines
where to resume execution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphrl.state import ModuleOutput

logger = logging.getLogger(__name__)


def detect_progress(
    experiment_dir: Path,
    num_iterations: int,
) -> Tuple[int, int, Optional[ModuleOutput]]:
    """
    Detect the latest completed phase for pipeline resume.

    Iterations are 0-indexed (0 .. num_iterations-1).

    Scans ``iter_XXX/`` directories in reverse order and checks for
    well-known completion markers at each phase:

      - **SFT complete**: ``iter_XXX/sft_model/config.json`` exists
      - **TrajToSFT complete**: ``iter_XXX/sft_data/dataset_info.json`` exists
      - **RL complete**: ``iter_XXX/rl_model/config.json`` exists

    Returns:
        (start_iteration_idx, start_phase_idx, last_output)
    """

    for iter_idx in range(num_iterations - 1, -1, -1):
        iter_dir = experiment_dir / f"iter_{iter_idx:03d}"
        if not iter_dir.exists():
            continue

        # Phase 3 complete: SFT model ready
        sft_model_dir = iter_dir / "sft" / "sft_model"
        if (sft_model_dir / "config.json").exists():
            output = ModuleOutput(model_path=str(sft_model_dir))
            if iter_idx >= num_iterations - 1:
                logger.info(f"Pipeline already complete (all {num_iterations} iterations done)")
                return num_iterations, 0, output
            logger.info(f"Resuming after iteration {iter_idx} (SFT complete)")
            return iter_idx + 1, 0, output

        # Phase 2 complete: SFT data ready -> resume at SFT (phase index 2).
        # We require the ``.phase_done`` marker (written at the end of
        # TrajToSFTModule.launch()) rather than just ``dataset_info.json``
        # — the latter appears halfway through any reasoning post-step,
        # which would let an interrupted reasoning step look "done" and
        # silently skip the rest on resume.
        sft_data_dir = iter_dir / "traj_to_sft" / "sft_data"
        if (sft_data_dir / ".phase_done").exists():
            rl_model_dir = iter_dir / "rl" / "rl_model"
            model = str(rl_model_dir) if (rl_model_dir / "config.json").exists() else None
            output = ModuleOutput(
                model_path=model,
                data_paths={"sft_data": str(sft_data_dir)},
            )
            logger.info(f"Resuming iteration {iter_idx} at SFT phase (TrajToSFT complete)")
            return iter_idx, 2, output

        # Phase 1 complete: RL model ready -> resume at TrajToSFT (phase index 1)
        rl_model_dir = iter_dir / "rl" / "rl_model"
        if (rl_model_dir / "config.json").exists():
            graph_dir = iter_dir / "traj_to_sft" / "graph"
            trajs_dir = iter_dir / "trajs"
            output = ModuleOutput(
                model_path=str(rl_model_dir),
                data_paths={
                    "graph": str(graph_dir) if graph_dir.exists() else "",
                    "trajs": str(trajs_dir) if trajs_dir.exists() else "",
                },
            )
            logger.info(f"Resuming iteration {iter_idx} at TrajToSFT phase (RL complete)")
            return iter_idx, 1, output

    logger.info("No previous progress detected, starting from scratch")
    return 0, 0, None
