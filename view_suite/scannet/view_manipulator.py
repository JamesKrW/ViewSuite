from annotated_types import T
import numpy as np
from scipy.spatial.transform import Rotation as R
from view_suite.scannet.utils.pose_utils import (
    extrinsic_c2w_to_w2c,
    extrinsic_w2c_to_c2w,
    c2w_extrinsic_to_se3,
    w2c_extrinsic_to_se3,
    check_4x4,
    assert_rotation,
)


class ViewManipulator:
    """
    Camera pose controller that keeps the canonical state as camera-to-world (c2w).
    Movement is along the camera's own axes (no ground projection).

    Movement:
      - Forward/backward: along camera +Z (into the screen for your stack).
      - Right/left:       along camera +X (screen-right).
      - Screen up/down:   along camera ±Y (sign depends on image_y_down).

    Rotation:
      - Yaw:   rotate around WORLD up axis (pre-multiply in c2w).
      - Pitch: rotate around LOCAL camera right X (post-multiply in c2w).

    Discrete mode:
      If is_discrete=True, rotation is snapped so c2w intrinsic 'xyz' Euler
      angles are integer multiples of step_rotation_deg.

    Poses:
      - get_pose(mode='c2w'|'w2c'): returns extrinsic in requested convention.
      - get_world_se3 / set_world_se3: 6-DoF for c2w (camera absolute pose in world).
      - get_se3 / set_se3:             6-DoF for w2c (rarely needed; t is not camera center).
    """

    def __init__(
        self,
        step_translation: float = 0.5,
        step_rotation_deg: float = 30.0,
        world_up_axis: str = "Z",
        is_discrete: bool = False,
        is_snap_every_step: bool = True,
        image_y_down: bool = True,
    ):
        """
        Args:
            step_translation: translation step length in world units.
            step_rotation_deg: rotation step size in degrees.
            world_up_axis: 'Z' (ScanNet-style) or 'Y'.
            is_discrete: use fixed step sizes for actions.
            is_snap_every_step: if True, snap rotations to nearest multiples of
                step_rotation_deg after every rotation (Euler quantization).
                Only effective when is_discrete=True.
            image_y_down: if True, screen-up = camera (0,-1,0); else (0,+1,0).
        """
        self.step_t = float(step_translation)
        self.step_r_deg = float(step_rotation_deg)
        self.step_r = np.radians(self.step_r_deg)
        self.up_axis = world_up_axis.upper()
        assert self.up_axis in ("Z", "Y"), "world_up_axis must be 'Z' or 'Y'"
        self.is_discrete = bool(is_discrete)
        self.is_snap_every_step = bool(is_snap_every_step) and self.is_discrete
        self.image_y_down = bool(image_y_down)

        # Canonical pose: camera-to-world
        self.c2w = np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Initialization / getters / setters
    # -------------------------------------------------------------------------
    def reset(self, initial_extrinsic_c2w: np.ndarray | None = None) -> np.ndarray:
        """
        Reset to identity or provided camera-to-world (4x4). In discrete mode,
        the rotation is snapped to multiples of step_rotation_deg.
        """
        if initial_extrinsic_c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            check_4x4(initial_extrinsic_c2w)
            self.c2w = initial_extrinsic_c2w.astype(np.float64)
        if self.is_snap_every_step:
            self._snap_rotation_in_place()
        return self.get_pose(mode="c2w")

    def get_pose(self, mode: str = "c2w") -> np.ndarray:
        """
        Return current extrinsic:
          - mode='c2w': camera-to-world (4x4)
          - mode='w2c': world-to-camera (4x4)
        """
        if mode == "c2w":
            return self.c2w.copy()
        elif mode == "w2c":
            return extrinsic_c2w_to_w2c(self.c2w)
        else:
            raise ValueError("mode must be 'c2w' or 'w2c'")

    # -------------------------------------------------------------------------
    # Discrete action API
    # -------------------------------------------------------------------------
    # step(): add 't'/'g' for on-screen CCW/CW (content rotation)
    def step(self, action: str) -> np.ndarray:
        a = action.strip().lower()
        if   a == "w": self.move_forward(+self.step_t)
        elif a == "s": self.move_forward(-self.step_t)
        elif a == "a": self.move_right(-self.step_t)
        elif a == "d": self.move_right(+self.step_t)
        elif a == "y": self.move_screen_up(+self.step_t)
        elif a == "h": self.move_screen_up(-self.step_t)
        elif a == "q": self.yaw_camera(-self.step_r)  # turn left (local +Y)
        elif a == "e": self.yaw_camera(+self.step_r)  # turn right
        elif a == "r":
            ang = (+self.step_r) if self.image_y_down else (-self.step_r)  # look up
            self.pitch_camera(ang)
        elif a == "f":
            ang = (-self.step_r) if self.image_y_down else (+self.step_r)  # look down
            self.pitch_camera(ang)
        elif a == "t":
            # on-screen CCW (rotate content CCW) => camera roll negative
            self.roll_camera(-self.step_r)
        elif a == "g":
            # on-screen CW (rotate content CW) => camera roll positive
            self.roll_camera(+self.step_r)
        else:
            raise ValueError(f"Unsupported action: {action}")
        return self.get_pose(mode="c2w")


    # -------------------------------------------------------------------------
    # Movements (no ground projection) in WORLD via c2w
    # -------------------------------------------------------------------------
    def move_forward(self, distance: float):
        """Translate along camera +Z by `distance` (world = c2w[:3,:3] @ [0,0,1])."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])  # camera +Z in world
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    def move_right(self, distance: float):
        """Translate along screen-right (camera +X) by `distance`."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([1.0, 0.0, 0.0])  # camera +X in world
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    def move_screen_up(self, distance: float):
        """
        Translate along screen-up by `distance`.
        If image_y_down=True (OpenCV-like), screen-up = camera (0,-1,0);
        otherwise screen-up = camera (0,+1,0).
        """
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        cam_up = np.array([0.0, -1.0, 0.0]) if self.image_y_down else np.array([0.0, 1.0, 0.0])
        dir_world = R_c2w @ cam_up
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    # -------------------------------------------------------------------------
    # Rotations (about camera center) in c2w
    # -------------------------------------------------------------------------
    def yaw_camera(self, angle_rad: float):
        """
        Yaw around the camera's local +Y axis by angle_rad, about the camera center.
        Implemented in c2w as: R_c2w' = R_c2w @ R_y(angle).
        """
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("y", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_snap_every_step:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)

    def pitch_camera(self, angle_rad: float):
        """
        Pitch around the camera's local +X axis by angle_rad, about the camera center.
        Implemented in c2w as: R_c2w' = R_c2w @ R_x(angle).
        """
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("x", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_snap_every_step:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)

        # Roll around camera local +Z (view axis)
    def roll_camera(self, angle_rad: float):
        """
        Roll around the camera's local +Z axis by angle_rad, about the camera center.
        Implemented in c2w as: R_c2w' = R_c2w @ R_z(angle)

        Note on on-screen rotation:
        - Positive camera roll (angle_rad > 0) makes the *content* appear to rotate CW.
        - Negative camera roll makes the content appear to rotate CCW.
        """
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("z", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_snap_every_step:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)


    # -------------------------------------------------------------------------
    # 6-DoF conversions
    # -------------------------------------------------------------------------
    # ---- c2w (camera->world) absolute pose, single API with degrees flag ----
    def get_se3(self, degrees: bool = True) -> np.ndarray:
        """
        Return the camera-to-world pose as SE(3) = [cx, cy, cz, rx, ry, rz].
        - Translation (cx,cy,cz) is camera center in world coordinates.
        - Rotation (rx,ry,rz) are intrinsic 'xyz' Euler angles of the c2w rotation.
        - If degrees=True, angles are in degrees; otherwise radians.
        """
        R_c2w = self.c2w[:3, :3]
        C_world = self.c2w[:3, 3]
        eul = R.from_matrix(R_c2w).as_euler('xyz', degrees=degrees)
        return np.concatenate([C_world.astype(np.float64), eul.astype(np.float64)])


    def set_se3(self, pose6: np.ndarray, degrees: bool = True):
        """
        Set the camera-to-world pose from SE(3) = [cx, cy, cz, rx, ry, rz].
        - Translation (cx,cy,cz) is camera center in world coordinates.
        - Rotation (rx,ry,rz) are intrinsic 'xyz' Euler angles of the c2w rotation.
        - If degrees=True, input angles are interpreted as degrees; otherwise radians.
        - Discrete mode (if enabled) snaps the resulting rotation to step_rotation_deg.
        """
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
        if pose6.shape[0] != 6:
            raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")
        C_world = pose6[:3]
        angles = pose6[3:]
        R_c2w = R.from_euler('xyz', angles, degrees=degrees).as_matrix()
        # optional snapping
        if self.is_snap_every_step:
            e = R.from_matrix(R_c2w).as_euler('xyz', degrees=False)
            e = self.step_r * np.round(e / self.step_r)
            R_c2w = R.from_euler('xyz', e, degrees=False).as_matrix()
        # compose c2w
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_c2w
        M[:3, 3] = C_world
        self.c2w = M


    # -------------------------------------------------------------------------
    # Internal helpers (c2w)
    # -------------------------------------------------------------------------
    @staticmethod
    def _Rc_t_from_c2w(M: np.ndarray):
        """Extract (R_c2w, C_world) from a 4x4 c2w matrix."""
        check_4x4(M)
        R_c2w = M[:3, :3]
        t = M[:3, 3]
        assert_rotation(R_c2w)
        return R_c2w, t

    @staticmethod
    def _compose_c2w(R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
        """Compose a 4x4 c2w from rotation and camera center."""
        assert_rotation(R_c2w)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_c2w
        M[:3, 3] = C_world.astype(np.float64)
        return M

    def _translate_camera_center(self, C_world: np.ndarray, R_c2w: np.ndarray, delta_world: np.ndarray):
        """
        Move camera center by delta in world coordinates, preserving orientation.
        """
        C_new = C_world + delta_world
        self.c2w = self._compose_c2w(R_c2w, C_new)

    # ----- discrete snapping on c2w -----
    def _snap_angles(self, eul_xyz_rad: np.ndarray) -> np.ndarray:
        """Snap each Euler angle (rad) to nearest multiple of step_r."""
        return self.step_r * np.round(eul_xyz_rad / self.step_r)

    def _snap_rotation_matrix_c2w(self, R_c2w: np.ndarray) -> np.ndarray:
        """Snap a c2w rotation matrix via Euler 'xyz' rounding."""
        e = R.from_matrix(R_c2w).as_euler('xyz', degrees=False)
        e = self._snap_angles(e)
        return R.from_euler('xyz', e, degrees=False).as_matrix()

    def _snap_rotation_in_place(self):
        """
        Snap current rotation (on c2w) while preserving camera center.
        """
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_snapped = self._snap_rotation_matrix_c2w(R_c2w)
        self.c2w = self._compose_c2w(R_snapped, C_world)
