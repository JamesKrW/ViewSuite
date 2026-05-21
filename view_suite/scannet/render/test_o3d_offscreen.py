#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Open3D Offscreen Rendering Test (v0.19, GPU/EGL)
- Uses EGL headless backend (OpenGL on GPU)
- Renders a single unlit box (no lighting/PBR to avoid API variance)
- Saves color.png and depth.png
- Exits 0 on success (both images written), 1 otherwise
"""

import os
import sys
import argparse
import numpy as np

# --- Make EGL headless robust in typical containers/servers ---
os.environ.setdefault("OPEN3D_HEADLESS_RENDERING", "1")
os.environ.setdefault("OPEN3D_RENDERING_BACKEND", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", "/usr/share/glvnd/egl_vendor.d")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_DIRS", "/usr/share/glvnd/glx_vendor.d")
if not os.environ.get("XDG_RUNTIME_DIR"):
    xdg = f"/tmp/{os.environ.get('USER', 'root')}-runtime"
    os.makedirs(xdg, exist_ok=True)
    os.chmod(xdg, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = xdg

import open3d as o3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="o3d_offscreen_out", help="Output directory")
    ap.add_argument("--w", type=int, default=640, help="Render width")
    ap.add_argument("--h", type=int, default=480, help="Render height")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    color_path = os.path.join(args.out, "color.png")
    depth_path = os.path.join(args.out, "depth.png")

    print("=== Open3D Offscreen Minimal Test (GPU/EGL) ===")
    print(f"Open3D version: {o3d.__version__}")
    print(f"DISPLAY={os.environ.get('DISPLAY')}")
    print(f"Output dir: {os.path.abspath(args.out)}")
    print(f"Target size: {args.w}x{args.h}")

    # Optional CUDA availability print (independent of rendering backend)
    try:
        import open3d.core as o3c  # present in 0.19
        print(f"Open3D CUDA available (o3d.core.cuda): {o3c.cuda.is_available()}")
    except Exception:
        pass

    # Create offscreen renderer (uses GPU via EGL/OpenGL when configured)
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(args.w, args.h)
    except Exception as e:
        print("[ERROR] Failed to create OffscreenRenderer (EGL/GL unavailable?).")
        print("Exception:", e)
        sys.exit(1)

    try:
        renderer.scene.set_background([0.05, 0.05, 0.05, 1.0])

        # Geometry: a simple box
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
        mesh.compute_vertex_normals()

        # Unlit material avoids lights/PBR fields across versions
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = (0.7, 0.8, 1.0, 1.0)

        # Add to scene
        renderer.scene.add_geometry("box", mesh, mat)

        # Camera: look at AABB center from an offset position
        aabb = mesh.get_axis_aligned_bounding_box()
        center = aabb.get_center()  # returns a 3-float vector-like
        eye = [center[0] + 1.8, center[1] + 1.6, center[2] + 1.2]
        up = [0.0, 0.0, 1.0]
        renderer.scene.camera.look_at(center, eye, up)

        # Render color
        color = renderer.render_to_image()
        ok_color = o3d.io.write_image(color_path, color, quality=9)
        print(f"[Color] saved={ok_color} path={color_path}")

        # Render depth; save a normalized 16-bit PNG for quick inspection
        depth = renderer.render_to_depth_image()
        depth_np = np.asarray(depth)
        ok_depth = False
        if depth_np.size > 0:
            vis = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            m = float(np.max(vis)) if vis.size else 0.0
            if m > 0.0:
                vis = (vis / m) * 65535.0
            vis_img = o3d.geometry.Image(vis.astype(np.uint16))
            ok_depth = o3d.io.write_image(depth_path, vis_img)
        print(f"[Depth] saved={ok_depth} path={depth_path}")

        if ok_color and ok_depth:
            print("Minimal offscreen render succeeded (GPU via EGL/OpenGL).")
            sys.exit(0)
        else:
            print("Failed to write outputs.")
            sys.exit(1)

    except Exception as e:
        print("[ERROR] Exception during rendering:", repr(e))
        sys.exit(1)
    finally:
        try:
            renderer.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
