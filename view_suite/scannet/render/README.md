
##  Open3D Headless Rendering Setup

### Install System Dependencies

```bash
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip python3-venv \
  libegl1 libgl1 libglx0 libglvnd0 libxext6 libx11-6 libdrm2 libgbm1 \
  && ldconfig
```

These libraries provide `libEGL.so.1`, `libGL.so.1`, and related GLVND dependencies required by Open3D’s Filament backend.



### Run the Offscreen Rendering Test

```bash
python test_o3d_offscreen.py --out ./o3d_offscreen_out --w 800 --h 600
```

Expected output:

```
[Open3D INFO] EGL headless mode enabled.
EGL(1.5)
OpenGL(4.1)
[Color] Render time: ...
[Depth] Render time: ...
✅ Offscreen rendering appears to be working.
```

Output files:

* `color.png` — RGB image of the scene
* `depth.png` — 16-bit normalized depth visualization

