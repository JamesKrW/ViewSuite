# view_suite/envs/utils/pose_eval_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def wrap_deg(x: np.ndarray) -> np.ndarray:
    """Map any degree angle(s) to (-180, 180]."""
    x = np.asarray(x, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def geodesic_angle_deg(euler_xyz_deg_a: np.ndarray, euler_xyz_deg_b: np.ndarray) -> float:
    """
    Geodesic angle between two rotations given as Euler degrees (SciPy seq 'xyz').
    Returns angle in [0, 180].
    """
    Ra = R.from_euler('xyz', np.asarray(euler_xyz_deg_a, dtype=np.float64), degrees=True).as_matrix()
    Rb = R.from_euler('xyz', np.asarray(euler_xyz_deg_b, dtype=np.float64), degrees=True).as_matrix()
    Rrel = Ra @ Rb.T
    cos_theta = (np.trace(Rrel) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def smooth_closeness_score(error: float, threshold: float, *, steepness: float = 6.0) -> float:
    """Map error to [0,1] using a logistic curve centred on the threshold."""
    if error <= 1e-9:
        return 1.0
    if threshold <= 1e-9:
        return 0.0
    scaled = (error / threshold) - 1.0
    z = steepness * scaled
    if z >= 60.0:
        return 0.0
    if z <= -60.0:
        return 1.0
    score = 1.0 / (1.0 + math.exp(z))
    return float(max(0.0, min(1.0, score)))


def resolve_thresholds(
    item: Dict[str, Any],
    tol_trans_l2_m: Optional[float],
    tol_rot_l2_deg: Optional[float],
    default_trans: float = 0.5,
    default_rot: float = 45.0,
) -> Tuple[float, float]:
    """
    Determine thresholds with precedence:
      1) env config tol_trans_l2_m / tol_rot_l2_deg (if not None)
      2) item['success_criteria'] pose_distance_threshold / angle_threshold (if present)
      3) defaults
    """
    t_l2 = tol_trans_l2_m
    r_l2 = tol_rot_l2_deg

    if t_l2 is None or r_l2 is None:
        succ = item.get("success_criteria") or {}
        if t_l2 is None:
            t_l2 = float(succ.get("pose_distance_threshold", default_trans))
        if r_l2 is None:
            r_l2 = float(succ.get("angle_threshold", default_rot))

    if t_l2 is None:
        t_l2 = default_trans
    if r_l2 is None:
        r_l2 = default_rot

    return float(t_l2), float(r_l2)


def parse_multi_level_success_rate(spec: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    """
    Parse "1,60;1,90;2,60" -> [(1.0, 60.0), (1.0, 90.0), (2.0, 60.0)]
    Returns None if spec is None/empty.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    out: List[Tuple[float, float]] = []
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for p in parts:
        ab = [x.strip() for x in p.split(",") if x.strip()]
        if len(ab) != 2:
            raise ValueError(f"Invalid multi_level_success_rate entry: '{p}', expected 'meters,degrees'")
        m = float(ab[0])
        d = float(ab[1])
        out.append((m, d))

    out = sorted(set(out), key=lambda x: (x[0], x[1]))
    return out


def multi_level_success_flags(
    pos_err_m: float,
    ang_err_deg: float,
    levels: Optional[List[Tuple[float, float]]],
) -> Dict[str, bool]:
    """
    Build dict like:
      success_1m60degree: True/False
    """
    if not levels:
        return {}

    flags: Dict[str, bool] = {}
    for m, d in levels:
        key = f"success_{_fmt_level(m)}m{_fmt_level(d)}degree"
        flags[key] = (pos_err_m <= m + 1e-9) and (ang_err_deg <= d + 1e-9)
    return flags


def _fmt_level(x: float) -> str:
    """Make keys stable: 1.0->'1', 0.5->'0p5', 60.0->'60'"""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.6g}"  # compact
    return s.replace(".", "p").replace("-", "neg")


def resolve_thresholds_per_action_len(
    tol_per_action_len: str,
    action_len: int
) -> Tuple[float, float]:
    """
    Parse and resolve tolerances based on action length (integer).

    Mapping grammar (segments separated by ';'):

      1) Default segment (gap segment):
         "trans,rot"
         -> used for uncovered regions between/around ranges.

      2) Range segment:
         "a-b:trans,rot"
         -> inclusive range [a, b]

      3) Point segment (allowed):
         "a:trans,rot"
         -> treated as inclusive range [a, a]

    IMPORTANT CONSTRAINTS:
      - First and last segments must be defaults.
      - Between non-contiguous ranges, a default segment is required.
      - Contiguous integer ranges (e.g., [2,2] and [3,5] where 3 == 2+1) do NOT require
        a default between them.
      - Ranges must be strictly increasing and non-overlapping.

    Examples:

      A) Classic 3-bucket:
         "0.5,30;3-5:1,30;1,60"
         - action_len < 3        -> (0.5, 30)
         - 3 <= action_len <= 5  -> (1, 30)
         - action_len > 5        -> (1, 60)

      B) Allow point segment:
         "0.5,30;3:1,30;1,60"
         - action_len < 3        -> (0.5, 30)
         - action_len == 3       -> (1, 30)
         - action_len > 3        -> (1, 60)

      C) Contiguous integer ranges (no default needed between them):
         "0.25,15;2:0.5,30;3-5:1,30;1,60"
         - action_len < 2        -> (0.25, 15)
         - action_len == 2       -> (0.5, 30)
         - 3 <= action_len <= 5  -> (1, 30)
         - action_len > 5        -> (1, 60)

      D) Non-contiguous ranges require default in between:
         "1,30;2:0.8,20;0.9,25;5-7:1.2,40;1.5,60"
         - action_len < 2        -> (1, 30)
         - action_len == 2       -> (0.8, 20)
         - 2 < action_len < 5    -> (0.9, 25)   # gap: 3, 4
         - 5 <= action_len <= 7  -> (1.2, 40)
         - action_len > 7        -> (1.5, 60)
    """
    if not tol_per_action_len or not tol_per_action_len.strip():
        raise ValueError("tol_per_action_len is empty")

    raw_parts = [p.strip() for p in tol_per_action_len.split(";") if p.strip()]
    if not raw_parts:
        raise ValueError("tol_per_action_len has no valid segments")

    def parse_vals(vals: str) -> Tuple[float, float]:
        items = [x.strip() for x in vals.split(",") if x.strip()]
        if len(items) != 2:
            raise ValueError(f"Invalid tolerance values '{vals}', expected 'trans,rot'")
        return float(items[0]), float(items[1])

    def parse_range_spec(spec: str) -> Tuple[int, int]:
        spec = spec.strip()
        if "-" in spec:
            a, b = [x.strip() for x in spec.split("-", 1)]
            return int(a), int(b)
        # "a:..." means a single point [a, a]
        v = int(spec)
        return v, v

    # Parse into ordered segments (preserve order)
    # segment: ("default", (trans,rot)) or ("range", (start,end, (trans,rot)))
    segments: List[Tuple[str, object]] = []
    for part in raw_parts:
        if ":" in part:
            spec, vals = part.split(":", 1)
            start, end = parse_range_spec(spec)
            if start > end:
                raise ValueError(f"Invalid range '{spec}': start > end")
            segments.append(("range", (start, end, parse_vals(vals))))
        else:
            segments.append(("default", parse_vals(part)))

    # Validate: first and last must be defaults
    if segments[0][0] != "default" or segments[-1][0] != "default":
        raise ValueError(
            "Mapping must start and end with a default segment 'trans,rot'."
        )

    # Validate alternation with contiguous integer ranges exception
    # Collect ranges in order to check contiguity
    range_indices = [i for i, (kind, _) in enumerate(segments) if kind == "range"]

    for i in range(1, len(segments)):
        if segments[i][0] == segments[i-1][0]:
            # Two consecutive same types
            if segments[i][0] == "default":
                raise ValueError(
                    "Two consecutive default segments found. "
                    "Defaults should only appear between non-contiguous ranges."
                )
            else:
                # Two consecutive ranges - check if they are contiguous integers
                prev_range = segments[i-1][1]  # (start, end, (trans, rot))
                curr_range = segments[i][1]
                prev_end = prev_range[1]
                curr_start = curr_range[0]
                # For integers, contiguous means curr_start == prev_end + 1
                if curr_start != prev_end + 1:
                    raise ValueError(
                        f"Non-contiguous ranges without default in between. "
                        f"Range ending at {prev_end} and range starting at {curr_start} "
                        f"have a gap (integers {prev_end + 1} to {curr_start - 1}). "
                        f"Add a default segment between them."
                    )

    # Build the lookup structure: list of (start, end, trans_rot) for ranges
    # and defaults[i] for gaps before range[i], defaults[-1] for after last range
    defaults: List[Tuple[float, float]] = []
    ranges: List[Tuple[int, int, Tuple[float, float]]] = []

    # Track which default covers which gap
    # We need to associate defaults with the gaps they cover
    default_for_gap: List[Tuple[float, float]] = []  # default_for_gap[i] covers gap before ranges[i] or after ranges[i-1]

    last_was_range = False
    for kind, payload in segments:
        if kind == "default":
            defaults.append(payload)  # type: ignore[arg-type]
            last_was_range = False
        else:
            ranges.append(payload)  # type: ignore[arg-type]
            last_was_range = True

    if not ranges:
        if len(defaults) != 1:
            raise ValueError(
                f"Invalid mapping: without any ranges, mapping must contain exactly one default. "
                f"Got {len(defaults)} defaults."
            )
        return defaults[0]

    # Validate ranges are strictly increasing and non-overlapping
    prev_end: Optional[int] = None
    for idx, (start, end, _) in enumerate(ranges):
        if prev_end is not None and start <= prev_end:
            raise ValueError(
                f"Ranges overlap. Range #{idx} starts at {start} "
                f"but previous ends at {prev_end}. Require start > prev_end."
            )
        prev_end = end

    # Now resolve: find which segment covers action_len
    al = action_len

    # Build a mapping: for each position, determine what covers it
    # Strategy: iterate through segments in order, tracking current default
    current_default_idx = 0
    range_idx = 0

    # Re-parse segments to handle lookup correctly
    # We need to know: for a given action_len, which segment covers it?
    #
    # Approach: iterate through original segments order
    # - When we see a default, it covers the gap until the next range (or end)
    # - When we see a range, it covers [start, end]
    # - Contiguous ranges share no gap between them

    # Simpler approach: collect ranges in order, and for each gap, note the default
    # gaps: before first range, between non-contiguous ranges, after last range

    gap_defaults: List[Tuple[Optional[int], Optional[int], Tuple[float, float]]] = []
    # Each entry: (gap_start_exclusive, gap_end_exclusive, default_values)
    # gap_start_exclusive: the end of previous range (None means -inf)
    # gap_end_exclusive: the start of next range (None means +inf)

    current_default: Optional[Tuple[float, float]] = None
    prev_range_end: Optional[int] = None
    range_list = []

    for kind, payload in segments:
        if kind == "default":
            current_default = payload  # type: ignore[assignment]
        else:
            # It's a range
            start, end, trans_rot = payload  # type: ignore[misc]
            # Check if there's a gap between prev_range_end and start
            if prev_range_end is None:
                # First range: gap is (-inf, start)
                if current_default is not None:
                    gap_defaults.append((None, start, current_default))
            elif start > prev_range_end + 1:
                # There's a gap: (prev_range_end, start)
                if current_default is not None:
                    gap_defaults.append((prev_range_end, start, current_default))
            # else: contiguous, no gap

            range_list.append((start, end, trans_rot))
            prev_range_end = end
            current_default = None  # Reset after using

    # After all ranges, the remaining default covers (prev_range_end, +inf)
    # Find the last default in segments
    last_default = None
    for kind, payload in reversed(segments):
        if kind == "default":
            last_default = payload
            break
    if last_default is not None and prev_range_end is not None:
        gap_defaults.append((prev_range_end, None, last_default))  # type: ignore[arg-type]

    # Now lookup action_len
    # First check ranges
    for start, end, trans_rot in range_list:
        if start <= al <= end:
            return trans_rot

    # Check gaps
    for gap_start, gap_end, default_val in gap_defaults:
        in_gap = True
        if gap_start is not None and al <= gap_start:
            in_gap = False
        if gap_end is not None and al >= gap_end:
            in_gap = False
        if in_gap:
            return default_val

    # Fallback (should not reach here if config is correct)
    raise ValueError(f"No matching segment for action_len={al}")