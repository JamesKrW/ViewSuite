#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def calc_format_correct_rate(folder_path: str) -> float:
    """
    Calculate format correct rate for all rollouts in a folder.

    Args:
        folder_path: Path to folder containing rollout subfolders

    Returns:
        Format correct rate (0.0 to 1.0)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist", file=sys.stderr)
        sys.exit(1)

    total = 0
    correct = 0

    for subfolder in folder.iterdir():
        if not subfolder.is_dir():
            continue

        metrics_file = subfolder / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            cumulative_reward = metrics.get("cumulative_reward", 0)
            total += 1

            if cumulative_reward > 0:
                correct += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading {metrics_file}: {e}", file=sys.stderr)
            continue

    if total == 0:
        print("Error: No valid metrics.json files found", file=sys.stderr)
        return 0.0

    rate = correct / total
    return rate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder_path>", file=sys.stderr)
        sys.exit(1)

    folder_path = sys.argv[1]
    rate = calc_format_correct_rate(folder_path)
    print(rate)
