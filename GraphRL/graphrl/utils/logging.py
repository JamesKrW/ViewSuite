"""
Logging utilities for GraphRL.

Sets up dual logging: console output + file logging to the experiment directory.
"""

import logging
import sys
from pathlib import Path


def setup_logging(experiment_dir: Path, log_filename: str = "pipeline.log") -> None:
    """
    Configure the root logger with both console and file handlers.

    Args:
        experiment_dir: Directory where the log file will be written.
        log_filename: Name of the log file.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_file = experiment_dir / log_filename
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)
