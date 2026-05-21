"""
Visibility computation: determines which mesh vertices are visible from a camera pose.

Uses Open3D's GPU OffscreenRenderer for fast depth rendering, then compares
rendered depth against projected vertex depth to classify vertex visibility.

Note: render_to_depth_image(z_in_view_space=True) returns *ray distance*
(Euclidean distance along the ray), NOT camera-frame z.  We convert to z
via:  z = ray_distance * ray_dir_z, where ray_dir_z is the z-component
of the normalised camera-frame ray direction per pixel.

Typical usage:
    vc = VisibilityComputer("scene.ply", K_3x3, width=512, height=512)
    visible = vc.get_visible_vertex_indices(c2w_4x4)  # returns set of int
    vc.total_vertices  # total vertex count for coverage ratio
"""

from __future__ import annotations

from typing import Set

import numpy as np
import open3d as o3d

from view_suite.scannet.render.mesh_render import MeshRenderer


class VisibilityComputer:
    """
    Computes visible mesh-vertex indices from a given camera pose.

    Uses GPU-accelerated depth rendering (~12ms/pose) instead of CPU
    raycasting (~3200ms/pose).

    Lifecycle:
      1. Instantiate once per scene (loads mesh, builds GPU renderer).
      2. Call get_visible_vertex_indices() for each camera pose.
      3. Results are independent — safe to call repeatedly.
    """

    # Depth tolerance (metres) for the vertex-vs-surface depth test.
    DEPTH_EPSILON = 0.05
    # Near-plane clipping: vertices closer than this are ignored.
    NEAR_CLIP = 0.01

    def __init__(
        self,
        mesh_path: str,
        intrinsics_3x3: np.ndarray,
        width: int = 512,
        height: int = 512,
    ):
        """
        Args:
            mesh_path:      Path to a .ply triangle mesh file.
            intrinsics_3x3: 3x3 camera intrinsic matrix (fx, fy, cx, cy).
            width:          Image width in pixels.
            height:         Image height in pixels.
        """
        self.width = width
        self.height = height
        self.K_raw = np.asarray(intrinsics_3x3, dtype=np.float64)
        assert self.K_raw.shape == (3, 3), f"Expected 3x3 intrinsics, got {self.K_raw.shape}"

        # --- Load mesh ---
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh at {mesh_path} has no vertices.")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)  # (N, 3)
        self.total_vertices = len(self.vertices)

        # --- Apply letterbox scaling (same as MeshRenderer) ---
        # MeshRenderer scales intrinsics to fit the target resolution while
        # preserving field-of-view.  We must use the same scaled K for both
        # depth rendering and vertex projection so the depth values match.
        fx, fy, cx, cy, _ = MeshRenderer._scale_K_with_letterbox(
            self.K_raw, width, height
        )
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # --- GPU offscreen renderer for depth ---
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"  # unlit = fastest, no lighting needed for depth
        self._renderer.scene.add_geometry("mesh", mesh, mat)
        self._intr = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        # --- Precompute per-pixel ray_dir_z for depth conversion ---
        # render_to_depth_image(z_in_view_space=True) returns ray *distance*,
        # not camera-frame z.  Convert: z = ray_distance * ray_dir_z.
        self._ray_dir_z = self._precompute_ray_dir_z()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_visible_vertex_indices(self, c2w_4x4: np.ndarray) -> Set[int]:
        """
        Return the set of mesh vertex indices visible from *c2w_4x4*.

        Steps:
          1. Render depth on GPU → per-pixel z-depth map.
          2. Project all mesh vertices to the camera image plane.
          3. For in-frustum vertices, compare projected z with rendered z.
          4. Vertices passing the depth test are considered visible.
        """
        c2w = np.asarray(c2w_4x4, dtype=np.float64)
        w2c = np.linalg.inv(c2w)

        # 1. GPU depth rendering → z-depth map
        z_depth = self._render_z_depth(w2c)  # (H, W), inf where no surface

        # 2. Project vertices into camera frame (vectorised)
        R = w2c[:3, :3]  # (3, 3)
        t = w2c[:3, 3]   # (3,)
        verts_cam = (R @ self.vertices.T).T + t  # (N, 3)
        z = verts_cam[:, 2]

        # 3. Filter: in front of camera
        mask_front = z > self.NEAR_CLIP
        idx_front = np.nonzero(mask_front)[0]
        verts_front = verts_cam[mask_front]
        z_front = z[mask_front]

        # 4. Project to pixel coordinates using SCALED intrinsics
        proj = (self.K @ verts_front.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        # 5. Filter: within image bounds
        mask_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        idx_bounds = idx_front[mask_bounds]
        u_int = u[mask_bounds].astype(np.int32)
        v_int = v[mask_bounds].astype(np.int32)
        z_verts = z_front[mask_bounds]

        np.clip(u_int, 0, self.width - 1, out=u_int)
        np.clip(v_int, 0, self.height - 1, out=v_int)

        # 6. Depth test: vertex z vs. rendered surface z
        rendered_z = z_depth[v_int, u_int]
        mask_visible = np.abs(z_verts - rendered_z) < self.DEPTH_EPSILON

        visible_indices = idx_bounds[mask_visible]
        return set(visible_indices.tolist())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_ray_dir_z(self) -> np.ndarray:
        """
        Compute the z-component of the normalised camera-frame ray direction
        for each pixel.  Used to convert GPU ray-distance to z-depth.
        """
        K_inv = np.linalg.inv(self.K)
        u_coords = np.arange(self.width, dtype=np.float64) + 0.5
        v_coords = np.arange(self.height, dtype=np.float64) + 0.5
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        pixel_hom = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=-1)
        ray_dirs = np.einsum("ij,hwj->hwi", K_inv, pixel_hom)
        ray_norms = np.linalg.norm(ray_dirs, axis=-1)
        return (ray_dirs[..., 2] / ray_norms).astype(np.float64)

    def _render_z_depth(self, w2c: np.ndarray) -> np.ndarray:
        """
        Render depth on GPU and convert from ray-distance to camera-frame z.

        Returns:
            z_depth: (H, W) float64 array. Inf where no surface hit.
        """
        self._renderer.setup_camera(self._intr, w2c)
        ray_distance = np.asarray(
            self._renderer.render_to_depth_image(z_in_view_space=True)
        ).astype(np.float64)
        return ray_distance * self._ray_dir_z
