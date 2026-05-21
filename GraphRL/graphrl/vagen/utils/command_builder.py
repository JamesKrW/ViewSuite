"""
VAGEN command builder.

Builds the command line for launching ``python3 -m vagen.main_ppo`` with
GraphRL's Hydra config directory as the config source.

Config architecture:
  - ``graphrl/configs/vagen_configs/`` is used as ``--config-path``, containing:
    - ``vagen_multiturn.yaml``: Hydra entry config with searchpath to VAGEN/verl
    - ``env_registry.yaml``: custom environment registry (overrides VAGEN's)
  - ``config["hydra_overrides"]``: flattened into Hydra CLI args (highest priority)
  - Controller-managed paths (model, checkpoint, rollout) are always appended last
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# GraphRL package root (graphrl/) — this file lives at graphrl/vagen/utils/
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# Default Hydra config directory for VAGEN
_DEFAULT_HYDRA_CONFIG_DIR = _PACKAGE_ROOT / "configs" / "vagen_configs"


def build_vagen_command(
    config: Dict[str, Any],
    model_path: str,
    output_dir: Path,
) -> List[str]:
    """
    Build the full command list for launching VAGEN PPO training.

    The command uses GraphRL's own Hydra config directory (which inherits
    from VAGEN via searchpath) instead of pointing directly to VAGEN's configs.

    ``config["hydra_overrides"]`` is flattened into Hydra CLI args.
    Everything else in config is for GraphRL internal use.

    The controller always injects these paths (appended last, wins over config):
      - ``actor_rollout_ref.model.path``
      - ``critic.model.path``
      - ``trainer.default_local_dir``
      - ``trainer.rollout_data_dir``
    """
    config_name = config.get("hydra_config_name", "vagen_multiturn")

    # Use GraphRL's Hydra config dir as the config source
    hydra_config_dir = config.get("hydra_config_dir")
    if hydra_config_dir:
        config_dir = Path(hydra_config_dir).resolve()
    else:
        config_dir = _DEFAULT_HYDRA_CONFIG_DIR

    cmd = [
        "python3",
        "-m",
        "vagen.main_ppo",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
    ]

    # Flatten hydra_overrides into CLI args
    hydra_overrides = config.get("hydra_overrides", {})
    if hydra_overrides:
        cmd.extend(_flatten(hydra_overrides, prefix=""))

    # Auto-inject training_steps so users don't have to set it in two places
    training_steps = config.get("training_steps")
    if training_steps is not None:
        cmd.append(f"trainer.total_training_steps={training_steps}")

    # Always inject controller-managed paths (appended last = highest priority)
    # Use absolute paths so VAGEN (which runs from a different CWD) writes to the right location.
    ckpt_dir = str(output_dir.resolve() / "verl_checkpoints")
    rollout_dir = str(output_dir.resolve() / "rollout_data")
    cmd.extend([
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.rollout_data_dir={rollout_dir}",
    ])

    # Inject WandB settings (always last so they override anything in hydra_overrides)
    project_name = config.get("_project_name", "graphrl")
    experiment_name = config.get("_experiment_name", "graphrl_pipeline")
    iter_num = config.get("_iter_num", 0)
    run_name = f"{experiment_name}_rl_iter{iter_num:03d}"
    cmd.extend([
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={run_name}",
        "trainer.logger=['console','wandb']",
    ])

    return cmd


# Keys that need Hydra's '+' prefix (new keys not in VAGEN's base config).
# Matches ViewSuite's convert_rl_rollout_to_sft convention.
_HYDRA_APPEND_KEY_PATTERNS = ["engine_kwargs", "eval_files"]


def _flatten(d: Union[Dict, Any], prefix: str) -> List[str]:
    """Recursively flatten a nested dict into Hydra CLI args."""
    if not isinstance(d, dict):
        # Add '+' prefix for keys that don't exist in VAGEN's base Hydra config
        if any(pattern in prefix for pattern in _HYDRA_APPEND_KEY_PATTERNS):
            return [f"+{prefix}={_format_value(d)}"]
        return [f"{prefix}={_format_value(d)}"]

    args: List[str] = []
    for key, value in d.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        args.extend(_flatten(value, new_prefix))
    return args


def _format_value(value: Any) -> str:
    """Format a Python value for Hydra CLI."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return str(value).replace(" ", "")
    return str(value)
