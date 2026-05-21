#!/usr/bin/env python3
"""
Dynamic rollouts visualization server.

This server reads rollouts directly from the filesystem and serves them via API.
All required metadata (scene_id/sample_id/gt_action/gt_action_len/success_*) is read
directly from metrics.json (no dependency on viewsuite DATA_DIR/meta.json).

New:
  - action_len_intervals is a CLI input (default "2,5,8,11") and not hardcoded.
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)

# Global configuration
ROLLOUTS_DIR: Optional[Path] = None
CUSTOM_FILTER_KEYS: List[str] = []
ACTION_LEN_INTERVALS: List[int] = [2, 5, 8, 11]  # default, can be overridden by CLI


# -----------------------------
# Helpers
# -----------------------------
def parse_action_len_intervals(s: str) -> List[int]:
    """
    Parse "2,5,8,11" -> [2,5,8,11]
    Accepts commas or semicolons, ignores whitespace.
    Enforces strictly increasing positive ints.
    """
    if s is None:
        return [2, 5, 8, 11]
    s = str(s).strip()
    if not s:
        return [2, 5, 8, 11]

    # allow comma or semicolon
    parts = []
    for chunk in s.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    vals: List[int] = []
    for p in parts:
        try:
            vals.append(int(p))
        except Exception:
            raise ValueError(f"Invalid action_len_intervals item '{p}', expected int")

    if not vals:
        raise ValueError("action_len_intervals parsed empty")

    # strictly increasing
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            raise ValueError(f"action_len_intervals must be strictly increasing, got {vals}")

    # non-negative (allow 0 if you really want, but usually intervals >0)
    if vals[0] < 0:
        raise ValueError(f"action_len_intervals must be >= 0, got {vals}")

    return vals


def get_rollout_metadata(rollout_path: Path) -> Optional[Dict[str, Any]]:
    """Get metadata for a rollout without loading full messages/images."""
    metrics_file = rollout_path / "metrics.json"
    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Prefer top-level fields; fallback to infos[0] if needed
        scene_id = metrics.get("scene_id")
        sample_id = metrics.get("sample_id")
        gt_action_len = metrics.get("gt_action_len")

        infos = metrics.get("infos", [])
        if isinstance(infos, list) and infos:
            first_info = infos[0] if isinstance(infos[0], dict) else {}

            if scene_id is None:
                scene_id = first_info.get("scene_id")
            if sample_id is None:
                sample_id = first_info.get("sample_id")

            if gt_action_len is None:
                gt_action = first_info.get("gt_action")
                if isinstance(gt_action, list):
                    gt_action_len = len(gt_action)
                elif "gt_action_len" in first_info and isinstance(first_info["gt_action_len"], int):
                    gt_action_len = first_info["gt_action_len"]

        if gt_action_len is None:
            gt_action = metrics.get("gt_action")
            if isinstance(gt_action, list):
                gt_action_len = len(gt_action)

        success_flags = {
            k: v for k, v in metrics.items()
            if k.startswith('success_') and isinstance(v, bool)
        }

        result = {
            'rollout_id': rollout_path.name,
            'success': metrics.get('success', False),
            'cumulative_reward': metrics.get('cumulative_reward', 0),
            'num_turns': metrics.get('num_turns', 0),
            'scene_id': scene_id,
            'sample_id': sample_id,
            'gt_action_len': gt_action_len,
        }
        result.update(success_flags)
        return result

    except Exception as e:
        print(f"Error reading metadata for {rollout_path}: {e}")
        return None


def get_rollout_full(rollout_path: Path) -> Optional[Dict[str, Any]]:
    """Get full rollout data including messages and images."""
    metrics_file = rollout_path / "metrics.json"
    messages_file = rollout_path / "messages.json"
    images_dir = rollout_path / "images"

    if not metrics_file.exists() or not messages_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        with open(messages_file, 'r') as f:
            messages = json.load(f)

        images = {}
        if images_dir.exists():
            for img_file in sorted(images_dir.glob("*.png")):
                with open(img_file, 'rb') as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                images[img_file.name] = f"data:image/png;base64,{base64_data}"

        return {
            'messages': messages,
            'images': images,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error loading full rollout {rollout_path}: {e}")
        return None


def get_interval_label(length: Optional[int], intervals: List[int]) -> str:
    """Get interval label for a given action sequence length."""
    if length is None:
        return 'Unknown'

    for i, threshold in enumerate(intervals):
        if length <= threshold:
            lower = intervals[i - 1] + 1 if i > 0 else 0
            upper = threshold
            return f"[{lower},{upper}]"

    lower = intervals[-1] + 1
    return f"[{lower},+∞]"


def get_available_filter_metrics(rollouts_metadata: List[Dict[str, Any]]) -> List[str]:
    all_filter_keys = set()

    for rollout in rollouts_metadata:
        for key, value in rollout.items():
            if key.startswith('success') and isinstance(value, bool):
                all_filter_keys.add(key)
            elif key in CUSTOM_FILTER_KEYS and isinstance(value, bool):
                all_filter_keys.add(key)

    return sorted(all_filter_keys)


def filter_rollouts_by_metric(
    rollouts_metadata: List[Dict[str, Any]],
    filter_key: str,
    filter_value: str
) -> List[Dict[str, Any]]:
    filtered = []

    for rollout in rollouts_metadata:
        if filter_value == 'missing':
            if filter_key not in rollout:
                filtered.append(rollout)
        elif filter_value == 'true':
            if filter_key in rollout and rollout[filter_key] is True:
                filtered.append(rollout)
        elif filter_value == 'false':
            if filter_key in rollout and rollout[filter_key] is False:
                filtered.append(rollout)

    return filtered


def compute_filter_metric_stats(
    rollouts_metadata: List[Dict[str, Any]],
    filter_key: str
) -> Dict[str, Any]:
    intervals = ACTION_LEN_INTERVALS

    true_count = 0
    false_count = 0
    missing_count = 0

    for rollout in rollouts_metadata:
        if filter_key not in rollout:
            missing_count += 1
        elif rollout[filter_key] is True:
            true_count += 1
        else:
            false_count += 1

    total = len(rollouts_metadata)
    total_with_metric = true_count + false_count

    overall_stats = {
        'total': total,
        'true': true_count,
        'false': false_count,
        'missing': missing_count,
        'true_rate': true_count / total_with_metric if total_with_metric > 0 else 0,
        'false_rate': false_count / total_with_metric if total_with_metric > 0 else 0,
        'missing_rate': missing_count / total if total > 0 else 0
    }

    action_len_stats = defaultdict(lambda: {'total': 0, 'true': 0, 'false': 0, 'missing': 0})

    for rollout in rollouts_metadata:
        gt_action_len = rollout.get('gt_action_len')
        if gt_action_len is not None:
            interval_label = get_interval_label(gt_action_len, intervals)
            action_len_stats[interval_label]['total'] += 1

            if filter_key not in rollout:
                action_len_stats[interval_label]['missing'] += 1
            elif rollout[filter_key] is True:
                action_len_stats[interval_label]['true'] += 1
            else:
                action_len_stats[interval_label]['false'] += 1

    for _, stats in action_len_stats.items():
        total_with_metric = stats['true'] + stats['false']
        total_in_interval = stats['total']
        stats['true_rate'] = stats['true'] / total_with_metric if total_with_metric > 0 else 0
        stats['false_rate'] = stats['false'] / total_with_metric if total_with_metric > 0 else 0
        stats['missing_rate'] = stats['missing'] / total_in_interval if total_in_interval > 0 else 0

    scene_stats = defaultdict(lambda: {'total': 0, 'true': 0, 'false': 0, 'missing': 0})

    for rollout in rollouts_metadata:
        scene_id = rollout.get('scene_id')
        if scene_id:
            scene_stats[scene_id]['total'] += 1

            if filter_key not in rollout:
                scene_stats[scene_id]['missing'] += 1
            elif rollout[filter_key] is True:
                scene_stats[scene_id]['true'] += 1
            else:
                scene_stats[scene_id]['false'] += 1

    for _, stats in scene_stats.items():
        total_with_metric = stats['true'] + stats['false']
        total_in_scene = stats['total']
        stats['true_rate'] = stats['true'] / total_with_metric if total_with_metric > 0 else 0
        stats['false_rate'] = stats['false'] / total_with_metric if total_with_metric > 0 else 0
        stats['missing_rate'] = stats['missing'] / total_in_scene if total_in_scene > 0 else 0

    return {
        'filter_key': filter_key,
        'overall': overall_stats,
        'by_action_len': dict(action_len_stats),
        'by_scene': dict(scene_stats)
    }


def compute_tag_statistics(rollouts_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    intervals = ACTION_LEN_INTERVALS

    total = len(rollouts_metadata)
    success_count = sum(1 for r in rollouts_metadata if r.get('success', False))
    success_rate = success_count / total if total > 0 else 0

    scene_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for rollout in rollouts_metadata:
        scene_id = rollout.get('scene_id')
        if scene_id:
            scene_stats[scene_id]['total'] += 1
            if rollout.get('success', False):
                scene_stats[scene_id]['success'] += 1

    for _, stats in scene_stats.items():
        stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0

    action_len_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for rollout in rollouts_metadata:
        gt_action_len = rollout.get('gt_action_len')
        if gt_action_len is not None:
            interval_label = get_interval_label(gt_action_len, intervals)
            action_len_stats[interval_label]['total'] += 1
            if rollout.get('success', False):
                action_len_stats[interval_label]['success'] += 1

    for _, stats in action_len_stats.items():
        stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0

    all_success_keys = set()
    for rollout in rollouts_metadata:
        for key in rollout.keys():
            if key.startswith('success_') and key != 'success':
                all_success_keys.add(key)

    success_metrics_stats = {}
    for success_key in sorted(all_success_keys):
        true_count = 0
        false_count = 0
        missing_count = 0

        for rollout in rollouts_metadata:
            if success_key in rollout:
                if rollout[success_key]:
                    true_count += 1
                else:
                    false_count += 1
            else:
                missing_count += 1

        total_with_metric = true_count + false_count
        overall_rate = true_count / total_with_metric if total_with_metric > 0 else 0

        action_len_success_stats = defaultdict(lambda: {'total': 0, 'true': 0, 'false': 0, 'missing': 0})

        for rollout in rollouts_metadata:
            gt_action_len = rollout.get('gt_action_len')
            if gt_action_len is not None:
                interval_label = get_interval_label(gt_action_len, intervals)
                action_len_success_stats[interval_label]['total'] += 1

                if success_key in rollout:
                    if rollout[success_key]:
                        action_len_success_stats[interval_label]['true'] += 1
                    else:
                        action_len_success_stats[interval_label]['false'] += 1
                else:
                    action_len_success_stats[interval_label]['missing'] += 1

        for _, stats in action_len_success_stats.items():
            twm = stats['true'] + stats['false']
            stats['success_rate'] = stats['true'] / twm if twm > 0 else 0

        success_metrics_stats[success_key] = {
            'overall': {
                'total': total,
                'true': true_count,
                'false': false_count,
                'missing': missing_count,
                'success_rate': overall_rate
            },
            'by_action_len': dict(action_len_success_stats)
        }

    return {
        'overall': {
            'total': total,
            'success': success_count,
            'success_rate': success_rate
        },
        'by_scene': dict(scene_stats),
        'by_action_len': dict(action_len_stats),
        'success_metrics': success_metrics_stats
    }


# -----------------------------
# API Routes
# -----------------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/tags')
def get_tags():
    tags_data = {}

    for model_dir in sorted(ROLLOUTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        tags_data[model_name] = []

        for tag_dir in sorted(model_dir.iterdir()):
            if not tag_dir.is_dir():
                continue

            tag_name = tag_dir.name
            rollout_count = sum(
                1 for d in tag_dir.iterdir()
                if d.is_dir() and (d / "metrics.json").exists()
            )

            tags_data[model_name].append({
                'name': tag_name,
                'rollout_count': rollout_count
            })

    return jsonify(tags_data)


@app.route('/api/tags/<model>/<tag>/rollouts')
def get_tag_rollouts(model, tag):
    tag_path = ROLLOUTS_DIR / model / tag
    if not tag_path.exists():
        return jsonify({'error': 'Tag not found'}), 404

    rollouts_metadata = []
    for rollout_dir in sorted(tag_path.iterdir()):
        if not rollout_dir.is_dir():
            continue
        metadata = get_rollout_metadata(rollout_dir)
        if metadata:
            rollouts_metadata.append(metadata)

    return jsonify(rollouts_metadata)


@app.route('/api/tags/<model>/<tag>/stats')
def get_tag_stats(model, tag):
    tag_path = ROLLOUTS_DIR / model / tag
    if not tag_path.exists():
        return jsonify({'error': 'Tag not found'}), 404

    rollouts_metadata = []
    for rollout_dir in sorted(tag_path.iterdir()):
        if not rollout_dir.is_dir():
            continue
        metadata = get_rollout_metadata(rollout_dir)
        if metadata:
            rollouts_metadata.append(metadata)

    stats = compute_tag_statistics(rollouts_metadata)
    return jsonify(stats)


@app.route('/api/tags/<model>/<tag>/filter_metrics')
def get_filter_metrics(model, tag):
    tag_path = ROLLOUTS_DIR / model / tag
    if not tag_path.exists():
        return jsonify({'error': 'Tag not found'}), 404

    rollouts_metadata = []
    for rollout_dir in sorted(tag_path.iterdir()):
        if not rollout_dir.is_dir():
            continue
        metadata = get_rollout_metadata(rollout_dir)
        if metadata:
            rollouts_metadata.append(metadata)

    filter_metrics = get_available_filter_metrics(rollouts_metadata)
    return jsonify({'filter_metrics': filter_metrics})


@app.route('/api/tags/<model>/<tag>/filter')
def get_filtered_rollouts(model, tag):
    tag_path = ROLLOUTS_DIR / model / tag
    if not tag_path.exists():
        return jsonify({'error': 'Tag not found'}), 404

    filter_key = request.args.get('filter_key')
    filter_value = request.args.get('filter_value')

    if not filter_key:
        return jsonify({'error': 'filter_key parameter is required'}), 400

    rollouts_metadata = []
    for rollout_dir in sorted(tag_path.iterdir()):
        if not rollout_dir.is_dir():
            continue
        metadata = get_rollout_metadata(rollout_dir)
        if metadata:
            rollouts_metadata.append(metadata)

    stats = compute_filter_metric_stats(rollouts_metadata, filter_key)

    if filter_value:
        filtered_rollouts = filter_rollouts_by_metric(rollouts_metadata, filter_key, filter_value)
    else:
        filtered_rollouts = rollouts_metadata

    return jsonify({
        'filter_key': filter_key,
        'filter_value': filter_value,
        'stats': stats,
        'rollouts_count': len(filtered_rollouts),
        'rollouts': filtered_rollouts
    })


@app.route('/api/rollouts/<model>/<tag>/<rollout_id>')
def get_rollout(model, tag, rollout_id):
    rollout_path = ROLLOUTS_DIR / model / tag / rollout_id
    if not rollout_path.exists():
        return jsonify({'error': 'Rollout not found'}), 404

    metadata = get_rollout_metadata(rollout_path)
    if not metadata:
        return jsonify({'error': 'Could not load rollout metadata'}), 500

    full_data = get_rollout_full(rollout_path)
    if not full_data:
        return jsonify({'error': 'Could not load rollout data'}), 500

    result = {**metadata, **full_data}
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description='Rollouts Visualization Server')
    parser.add_argument(
        '--rollouts_dir',
        type=str,
        default='/root/projects/viewsuite/data/rollouts',
        help='Path to rollouts directory'
    )
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument(
        '--custom_filter_keys',
        type=str,
        default='',
        help='Semicolon-separated list of additional custom filter keys (e.g., "key1;key2;key3")'
    )
    parser.add_argument(
        '--action_len_intervals',
        type=str,
        default="2,5,8,11",
        help='Comma/semicolon-separated thresholds for gt_action_len bucketing, e.g. "2,5,8,11"'
    )

    args = parser.parse_args()

    global ROLLOUTS_DIR, CUSTOM_FILTER_KEYS, ACTION_LEN_INTERVALS
    ROLLOUTS_DIR = Path(args.rollouts_dir)

    if args.custom_filter_keys:
        CUSTOM_FILTER_KEYS = [k.strip() for k in args.custom_filter_keys.split(';') if k.strip()]
    else:
        CUSTOM_FILTER_KEYS = []

    try:
        ACTION_LEN_INTERVALS = parse_action_len_intervals(args.action_len_intervals)
    except Exception as e:
        print(f"Error: invalid --action_len_intervals: {e}")
        return

    if not ROLLOUTS_DIR.exists():
        print(f"Error: Rollouts directory does not exist: {ROLLOUTS_DIR}")
        return

    print(f"Rollouts directory: {ROLLOUTS_DIR}")
    print(f"Action length intervals: {ACTION_LEN_INTERVALS}")
    if CUSTOM_FILTER_KEYS:
        print(f"Custom filter keys: {CUSTOM_FILTER_KEYS}")
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Open your browser and navigate to: http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
