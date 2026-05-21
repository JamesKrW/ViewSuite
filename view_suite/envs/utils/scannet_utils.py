import numpy as np
from typing import List
def parse_get_view_arg_deg(arg: str):
    """
    Parse 'tx,ty,tz,rx,ry,rz' into a list[float] (angles in DEGREES).
    Returns None if malformed.
    """
    try:
        parts = [p.strip() for p in arg.split(",")]
        if len(parts) != 6:
            return None
        return [float(x) for x in parts]
    except Exception:
        return None
    
def ensure_K4x4(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        return K
    if K.shape == (3, 3):
        K4 = np.eye(4, dtype=np.float64)
        K4[:3, :3] = K
        return K4
    raise ValueError(f"camera_intrinsics must be 3x3 or 4x4, got {K.shape}")


def default_intrinsics() -> np.ndarray:
    """
    Default intrinsics for 512x512 images based on typical ScanNet data.
    These values represent a common camera calibration from ScanNet scenes.

    Returns:
        np.ndarray: 4x4 intrinsics matrix
    """
    fx = 462.073
    fy = 617.312
    cx = 255.326
    cy = 259.135
    K4 = np.array([[fx, 0.0, cx, 0.0],
                    [0.0, fy, cy, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return K4


def fallback_K() -> np.ndarray:
    fx = fy = 300.0
    cx = cy = 150.0
    K4 = np.array([[fx, 0.0, cx, 0.0],
                    [0.0, fy, cy, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return K4

def fmt_pose6_deg(p: List[float]) -> str:
    tx, ty, tz, rx, ry, rz = p
    return f"[tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}, rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°]"