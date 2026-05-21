import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List
# ================================
# Utilities
# ================================
def check_4x4(M: np.ndarray, name: str = "matrix"):
    """Validate a (4,4) homogeneous matrix."""
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(M)}")
    if M.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {M.shape}")


def assert_rotation(Rm: np.ndarray, name: str = "R", atol: float = 1e-6):
    """Basic orthonormality check for a rotation matrix."""
    if Rm.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {Rm.shape}")
    should_be_I = Rm.T @ Rm
    if not np.allclose(should_be_I, np.eye(3), atol=atol):
        raise ValueError(f"{name} is not orthonormal within atol={atol}")
    if not np.isclose(np.linalg.det(Rm), 1.0, atol=atol):
        raise ValueError(f"{name} must have det=1, got det={np.linalg.det(Rm)}")
# ================================
# Matrix <-> Matrix conversions
# ================================
def extrinsic_c2w_to_w2c(camera_to_world: np.ndarray) -> np.ndarray:
    """
    Convert a camera-to-world (c2w) homogeneous matrix into a world-to-camera (w2c) extrinsic.
    Args:
        camera_to_world: (4,4) camera-to-world matrix
    Returns:
        (4,4) world-to-camera matrix
    """
    check_4x4(camera_to_world, name="camera_to_world")
    return np.linalg.inv(camera_to_world)


def extrinsic_w2c_to_c2w(world_to_camera: np.ndarray) -> np.ndarray:
    """
    Convert a world-to-camera (w2c) extrinsic into a camera-to-world (c2w) homogeneous matrix.
    Args:
        world_to_camera: (4,4) world-to-camera matrix
    Returns:
        (4,4) camera-to-world matrix
    """
    check_4x4(world_to_camera, name="world_to_camera")
    return np.linalg.inv(world_to_camera)


# ================================
# c2w Matrix <-> SE3 [t, r]
# ================================
def c2w_extrinsic_to_se3(camera_to_world: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Convert camera-to-world matrix into SE(3) 6-DoF parameterization.
    Args:
        camera_to_world: (4,4) camera-to-world matrix
        degrees: if True, return Euler angles in degrees; otherwise radians
    Returns:
        pose6: np.ndarray of shape (6,)
               [tx, ty, tz, rx, ry, rz] where t is camera center in world coords,
               and r are intrinsic 'xyz' Euler angles of the c2w rotation.
    """
    check_4x4(camera_to_world, name="camera_to_world")
    R_c2w = camera_to_world[:3, :3]
    t_c2w = camera_to_world[:3, 3]
    assert_rotation(R_c2w, name="R_c2w")

    eul = R.from_matrix(R_c2w).as_euler('xyz', degrees=degrees)
    return np.concatenate([t_c2w.astype(np.float64), eul.astype(np.float64)])


def c2w_se3_to_extrinsic(pose6: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Convert SE(3) parameterization into a camera-to-world matrix.
    Args:
        pose6: (6,) array-like [tx, ty, tz, rx, ry, rz]
               Euler angles are interpreted in degrees if degrees=True, else radians
        degrees: whether input angles are in degrees
    Returns:
        camera_to_world: (4,4) camera-to-world matrix
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.shape[0] != 6:
        raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")

    t = pose6[:3]
    r = pose6[3:]
    R_c2w = R.from_euler('xyz', r, degrees=degrees).as_matrix()

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_c2w
    M[:3, 3] = t
    return M


# ================================
# (Optional) w2c Matrix <-> SE3 [t, r] in camera frame
#   Provided for completeness; many pipelines only need c2w<->se3 above.
# ================================
def w2c_extrinsic_to_se3(world_to_camera: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Convert world-to-camera extrinsic into its SE(3) parameterization (of w2c).
    Note: translation here is the w2c 't' term (NOT camera center).
    Args:
        world_to_camera: (4,4) world-to-camera matrix
        degrees: if True, return Euler angles in degrees; otherwise radians
    Returns:
        (6,) [tx, ty, tz, rx, ry, rz] of the w2c transform
    """
    check_4x4(world_to_camera, name="world_to_camera")
    R_w2c = world_to_camera[:3, :3]
    t_w2c = world_to_camera[:3, 3]
    assert_rotation(R_w2c, name="R_w2c")

    eul = R.from_matrix(R_w2c).as_euler('xyz', degrees=degrees)
    return np.concatenate([t_w2c.astype(np.float64), eul.astype(np.float64)])


def w2c_se3_to_extrinsic(pose6: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Convert SE(3) parameterization (of w2c) back to a world-to-camera matrix.
    Args:
        pose6: (6,) [tx, ty, tz, rx, ry, rz] for the w2c transform
        degrees: whether input angles are in degrees
    Returns:
        (4,4) world-to-camera matrix
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.shape[0] != 6:
        raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")

    t = pose6[:3]
    r = pose6[3:]
    R_w2c = R.from_euler('xyz', r, degrees=degrees).as_matrix()

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_w2c
    M[:3, 3] = t
    return M

