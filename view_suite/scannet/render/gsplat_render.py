"""
GaussianSplatRenderer — renders a pretrained 3DGS PLY using gsplat.

Input PLY: standard 3DGS format (xyz + SH(deg=3) + scale + rot + opacity),
as produced by the gsplat MCMCStrategy (GaussianWorld/scannet_mcmc_1.5M_3dgs).

Interface matches MeshRenderer:
    render_image_from_cam_param(K, T_c2w, width=W, height=H) -> np.uint8 (H, W, 3)

Camera convention: OpenCV (+X right, +Y down, +Z forward), matching gsplat and ScanNet.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData

from .base_render import BaseRenderer

try:
    from gsplat import rasterization
except ImportError as e:
    rasterization = None
    _GSPLAT_IMPORT_ERROR = e
else:
    _GSPLAT_IMPORT_ERROR = None


class GaussianSplatRenderer(BaseRenderer):
    """Render a 3DGS PLY checkpoint via gsplat rasterization.

    Loads and caches one scene's gaussians on GPU. Subsequent calls to
    render_image_from_cam_param reuse the cached tensors.
    """

    def __init__(
        self,
        file_path: str,
        sh_degree: int = 3,
        rasterize_mode: str = "antialiased",
        device: Optional[str] = None,
    ):
        super().__init__(file_path)
        if rasterization is None:
            raise ImportError(
                f"gsplat is not installed (required by GaussianSplatRenderer): {_GSPLAT_IMPORT_ERROR}"
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"3DGS PLY not found: {file_path}")

        self.sh_degree = int(sh_degree)
        self.rasterize_mode = rasterize_mode
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._load_ply(file_path)
        print(
            f"GaussianSplatRenderer loaded: N={int(self.means.shape[0]):,} "
            f"sh_degree={self.sh_degree} device={self.device}"
        )

    def _load_ply(self, path: str) -> None:
        ply = PlyData.read(path)
        v = ply["vertex"].data
        N = len(v)

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        dc = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).astype(np.float32)

        f_rest_names = sorted(
            [n for n in v.dtype.names if n.startswith("f_rest_")],
            key=lambda s: int(s.split("_")[-1]),
        )
        K_sh = (self.sh_degree + 1) ** 2
        expected_rest = K_sh - 1
        f_rest_flat = np.stack(
            [v[n] for n in f_rest_names[: expected_rest * 3]], axis=-1
        ).astype(np.float32)  # (N, 3*K_rest)
        # 3DGS PLY stores features_rest as [N, 3, K_rest] row-major (after transpose).
        # Reshape [N, 3, K_rest] then transpose last two dims → [N, K_rest, 3].
        f_rest = f_rest_flat.reshape(N, 3, expected_rest).transpose(0, 2, 1)
        sh = np.concatenate([dc[:, None, :], f_rest], axis=1)  # (N, K_sh, 3)

        opacity_logit = v["opacity"].astype(np.float32)
        scale_log = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
        ).astype(np.float32)
        quat_raw = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)

        dev = self.device
        self.means = torch.from_numpy(xyz).to(dev)
        self.scales = torch.from_numpy(scale_log).to(dev).exp()
        quats = torch.from_numpy(quat_raw).to(dev)
        self.quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.opacities = torch.sigmoid(torch.from_numpy(opacity_logit).to(dev))
        self.sh_colors = torch.from_numpy(sh).to(dev)  # (N, K_sh, 3)

    @staticmethod
    def _to_K3x3(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=np.float64)
        if K.shape == (4, 4):
            return K[:3, :3]
        if K.shape == (3, 3):
            return K
        raise ValueError(f"camera_intrinsics must be 3x3 or 4x4, got {K.shape}")

    def render_image_from_cam_param(
        self,
        camera_intrinsics,
        camera_extrinsics,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """Render via gsplat. extrinsics is c2w (OpenCV)."""
        K3 = self._to_K3x3(np.asarray(camera_intrinsics, dtype=np.float64))
        T_c2w = np.asarray(camera_extrinsics, dtype=np.float64)
        w2c = np.linalg.inv(T_c2w).astype(np.float32)

        viewmats = torch.from_numpy(w2c)[None].to(self.device)  # (1,4,4)
        Ks = torch.from_numpy(K3.astype(np.float32))[None].to(self.device)  # (1,3,3)

        renders, _alphas, _info = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            colors=self.sh_colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            sh_degree=self.sh_degree,
            packed=False,
            rasterize_mode=self.rasterize_mode,
        )
        img = renders[0].clamp(0, 1).detach().cpu().numpy()  # (H, W, 3)
        img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
        return img_u8

    def release(self) -> None:
        """Free GPU memory. Called by handler when switching scenes."""
        for attr in ("means", "scales", "quats", "opacities", "sh_colors"):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
