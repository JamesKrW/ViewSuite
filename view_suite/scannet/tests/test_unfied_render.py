# test_unified_render.py
import os
import json
import time
import math
import asyncio
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import fire
import numpy as np
from PIL import Image

# Import your unified renderer
from view_suite.scannet.unified_renderer import UnifiedRender

# ------------------------- Helpers -------------------------

def ensure_dir(p: str) -> None:
    """Create directory if not exists."""
    os.makedirs(p, exist_ok=True)

def save_image(img: Image.Image, path: str) -> None:
    """Save PIL.Image to disk."""
    ensure_dir(os.path.dirname(path))
    img.save(path)

def make_dummy_intrinsics(width: int, height: int) -> np.ndarray:
    """Construct simple pinhole intrinsics with fx=fy=300 and center at image center."""
    fx = fy = 300.0
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def make_dummy_extrinsics(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """
    Build a 4x4 extrinsics (camera-to-world) from translation (meters)
    and Euler angles (radians) in XYZ order.
    """
    # Minimal rotation matrix from Euler XYZ
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rz * Ry * Rx (standard computer vision convention frequently varies;
    # here we just need a valid rotation for testing)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = np.array([tx, ty, tz], dtype=np.float32)
    return T

def monkey_patch_renderer_aliases(r: UnifiedRender) -> None:
    """
    Bridge internal/private method name mismatches so tests can run
    without editing the source.
    """
    # Only create aliases if missing
    if not hasattr(r, "_ensure_ply") and hasattr(r, "ensureply"):
        r._ensure_ply = r.ensureply  # type: ignore
    if not hasattr(r, "_ensure_local") and hasattr(r, "ensurelocal"):
        r._ensure_local = r.ensurelocal  # type: ignore
    if not hasattr(r, "_ensure_client") and hasattr(r, "ensureclient"):
        r._ensure_client = r.ensureclient  # type: ignore
    if not hasattr(r, "_to_pil") and hasattr(r, "topil"):
        r._to_pil = r.topil  # type: ignore

def build_tasks(width: int, height: int) -> List[Dict[str, Any]]:
    """Build a mixed list of two tasks: one 6DoF and one cam_param."""
    pose = [0.0, 0.0, 0.2,  # tx, ty, tz (meters)
            0.0, 0.15, 0.0]  # rx, ry, rz (radians)

    intr = make_dummy_intrinsics(width, height).tolist()
    extr = make_dummy_extrinsics(tx=0.0, ty=0.0, tz=0.2, rx=0.0, ry=0.0, rz=0.0).tolist()

    return [
        {"mode": "cam_param", "intrinsics": intr, "extrinsics": extr, "size": [width, height]},
    ]

# ------------------------- Core Test Flows -------------------------

async def run_local(scene_id: str,
                    scannet_root: str,
                    out_dir: str = "./out_local",
                    width: int = 320,
                    height: int = 240) -> None:
    """Test local backend: exercise all three public APIs and write images."""
    print("[Local] Initializing UnifiedRender...")
   
    r = UnifiedRender(
        render_backend="local",
        scannet_root=scannet_root,
        client_url=None,
        client_origin=None,
        scene_id=scene_id,
    )
    

    # Allow switching to the requested scene (ensures local cache not mixed)
    r.set_scene(scene_id)

    # 1) render_image_from_cam_param
    intr = make_dummy_intrinsics(width, height)
    extr = make_dummy_extrinsics(tx=0.0, ty=0.0, tz=0.2, rx=0.0, ry=0.0, rz=0.0)
    img1 = await r.render_image_from_cam_param(intr, extr, width, height)
    save_image(img1, os.path.join(out_dir, "local_cam_param.png"))
    print("[Local] Saved:", os.path.join(out_dir, "local_cam_param.png"))


    # 3) render_tasks
    tasks = build_tasks(width, height)
    imgs = await r.render_tasks(tasks)
    for i, im in enumerate(imgs):
        save_image(im, os.path.join(out_dir, f"local_tasks_{i}.png"))
    print("[Local] Saved:", os.path.join(out_dir, "local_tasks_0.png"), "and", os.path.join(out_dir, "local_tasks_1.png"))

    await r.close()
    print("[Local] Done.")

async def run_client(scene_id: str,
                     scannet_root: str,
                     url: str,
                     origin: Optional[str] = None,
                     out_dir: str = "./out_client",
                     width: int = 320,
                     height: int = 240) -> None:
    """Test client backend: exercise all three public APIs and write images."""
    print("[Client] Initializing UnifiedRender...")
    r = UnifiedRender(render_backend="client",scannet_root=None ,client_url=url, client_origin=origin, scene_id=scene_id)


    r.set_scene(scene_id)
    # Warm up client connection


    # 1) render_image_from_cam_param
    intr = make_dummy_intrinsics(width, height).tolist()
    extr = make_dummy_extrinsics(tx=0.0, ty=0.0, tz=0.2, rx=0.0, ry=0.0, rz=0.0).tolist()
    img1 = await r.render_image_from_cam_param(intr, extr, width, height)
    save_image(img1, os.path.join(out_dir, "client_cam_param.png"))
    print("[Client] Saved:", os.path.join(out_dir, "client_cam_param.png"))


    # 3) render_tasks
    tasks = build_tasks(width, height)
    imgs = await r.render_tasks(tasks)
    for i, im in enumerate(imgs):
        save_image(im, os.path.join(out_dir, f"client_tasks_{i}.png"))
    print("[Client] Saved:", os.path.join(out_dir, "client_tasks_0.png"), "and", os.path.join(out_dir, "client_tasks_1.png"))

    await r.close()
    print("[Client] Done.")

# ------------------------- Fire Entrypoints -------------------------

def test_local(
               scannet_root: str,
               scene_id: str="scene0011_00",
               out: str = "./out_local",
               width: int = 320,
               height: int = 240) -> None:
    """
    Launch local backend tests.

    Args:
        scene_id: e.g., "scene0011_00"
        scannet_root: path to ScanNet root containing /scans/<scene_id>/*
        out: output directory to save test images
        width/height: render size
    """
    asyncio.run(run_local(scene_id, scannet_root, out, width, height))

def test_client(
        url: str="ws://127.0.0.1:8766/render",
        scene_id: str="scene0011_00",
        origin: Optional[str] = None,
        scannet_root: Optional[str] = None,
        out: str = "./out_client",
        width: int = 320,
        height: int = 240) -> None:
    """
    Launch client backend tests.

    Args:
        scene_id: e.g., "scene0011_00"
        scannet_root: used by your server to resolve PLY (depending on server implementation)
        url: websocket url of render service, e.g., "ws://127.0.0.1:8766/render"
        origin: optional Origin header if your server validates it
        out: output directory to save test images
        width/height: render size
    """
    asyncio.run(run_client(scene_id, scannet_root, url, origin, out, width, height))

if __name__ == "__main__":
    fire.Fire({
        "test_local": test_local,
        "test_client": test_client,
    })


# test_client ws://127.0.0.1:8766/render --scene_id scene0011_00
# test_client http://0.0.0.0:8765 --scene_id scene0011_00