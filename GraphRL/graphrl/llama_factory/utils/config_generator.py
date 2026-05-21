"""
LLaMA-Factory config generator.

Generates the SFT training configuration YAML for ``llamafactory-cli train``.

Config architecture (same pattern as VAGEN):
  - ``config["hydra_overrides"]``: dict of LLaMA-Factory training parameters,
    written as a flat YAML config file.
  - Controller-managed paths (model, dataset_dir, output_dir) are injected
    automatically and override anything in hydra_overrides.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Directory containing default config files (ds_z3_config.json, etc.)
# This file lives at graphrl/llama_factory/utils/, so parents[2] = graphrl/.
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "llamafactory_configs"


def generate_sft_config(
    config: Dict[str, Any],
    model_path: str,
    dataset_dir: str,
    output_dir: Path,
) -> Path:
    """
    Generate a LLaMA-Factory SFT config YAML.

    Takes ``config["hydra_overrides"]`` as the base training config and
    injects controller-managed paths (model, dataset, output).

    For LoRA (``finetuning_type == "lora"``), the training output is
    redirected to ``output_dir/_lora_adapter/`` so that adapter files
    don't conflict with the final merged model in ``output_dir/``.

    Args:
        config: SFT module configuration dict.
        model_path: Path to the input model.
        dataset_dir: Path to the SFT dataset directory.
        output_dir: Output directory for the trained model.

    Returns:
        Path to the generated config YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "sft_config.yaml"

    # Start from hydra_overrides as base training config
    sft_config = dict(config.get("hydra_overrides", {}))

    # For LoRA, redirect training output to _lora_adapter/ subdirectory
    is_lora = sft_config.get("finetuning_type") == "lora"
    train_output_dir = output_dir / "_lora_adapter" if is_lora else output_dir
    train_output_dir.mkdir(parents=True, exist_ok=True)

    # Inject controller-managed paths (always override)
    sft_config["model_name_or_path"] = model_path
    sft_config["dataset_dir"] = str(dataset_dir)
    sft_config["output_dir"] = str(train_output_dir)

    # Inject WandB settings (always override so run names are consistent)
    project_name = config.get("_project_name", "graphrl")
    experiment_name = config.get("_experiment_name", "graphrl_pipeline")
    iter_num = config.get("_iter_num", 0)
    sft_config["report_to"] = "wandb"
    sft_config["run_name"] = f"{experiment_name}_sft_iter{iter_num:03d}"

    # Auto-detect dataset names from dataset_info.json
    dataset_info_path = Path(dataset_dir) / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        if dataset_info:
            sft_config["dataset"] = ",".join(dataset_info.keys())
            logger.info(f"Auto-detected datasets: {sft_config['dataset']}")

    # Resolve deepspeed config path (relative filenames -> absolute path)
    # A null/None value means deepspeed is explicitly disabled (e.g. for LoRA)
    if "deepspeed" in sft_config:
        if sft_config["deepspeed"] is None:
            del sft_config["deepspeed"]
        else:
            ds_path = Path(sft_config["deepspeed"]).expanduser()
            if not ds_path.is_absolute():
                # Resolve relative to the bundled config directory
                ds_path = _CONFIG_DIR / ds_path
            sft_config["deepspeed"] = str(ds_path)

    # Write config YAML
    with open(config_path, "w") as f:
        yaml.dump(sft_config, f, default_flow_style=False, sort_keys=False)

    # Handle resume from checkpoint (look in the actual training output dir)
    existing_checkpoints = sorted(
        [d for d in train_output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    ) if train_output_dir.exists() else []

    if existing_checkpoints:
        with open(config_path, "a") as f:
            f.write(f"\nresume_from_checkpoint: {existing_checkpoints[-1]}\n")
        logger.info(f"Resuming SFT from checkpoint: {existing_checkpoints[-1].name}")

    logger.info(f"Generated SFT config at {config_path}")
    return config_path


def generate_merge_config(
    model_path: str,
    adapter_dir: Path,
    export_dir: Path,
    template: str = "qwen2_vl",
) -> Path:
    """
    Generate a LLaMA-Factory export config YAML for merging a LoRA adapter.

    Used after LoRA training to merge the adapter weights back into the
    base model, producing a standard HuggingFace model directory.

    Args:
        model_path: Path to the base (pre-LoRA) model.
        adapter_dir: Path to the trained LoRA adapter directory.
        export_dir: Output directory for the merged model.
        template: Model template (must match the one used for training).

    Returns:
        Path to the generated merge config YAML file.
    """
    config_path = adapter_dir / "merge_config.yaml"

    merge_config = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": str(adapter_dir),
        "template": template,
        "finetuning_type": "lora",
        "export_dir": str(export_dir),
        "export_device": "auto",
        "export_size": 5,
    }

    with open(config_path, "w") as f:
        yaml.dump(merge_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated merge config at {config_path}")
    return config_path
