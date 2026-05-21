import os
import numpy as np
from PIL import Image

# Import your classes
#from point_cloud_render import PointCloudRenderer as Render   # or MeshRenderer
from view_suite.scannet.render.base_render import BaseRenderer
from view_suite.scannet.render.mesh_render import MeshRenderer as Render                 # or PointCloudRenderer
from view_suite.scannet.view_manipulator import ViewManipulator          # the gym we just wrote
from view_suite.scannet.utils.sens_reader import SensorData                  # if you want ScanNet intrinsics
from view_suite.scannet.utils.pose_utils import extrinsic_c2w_to_w2c
import fire
def render_current_view(renderer:BaseRenderer, gym:ViewManipulator, K, width, height):
    """
    Render the current camera view using the renderer and ViewManipulator's pose.
    Args:
        renderer: PointCloudRenderer or MeshRenderer
        gym: ViewManipulator object
        K (np.ndarray): Camera intrinsics (3x3)
        width (int), height (int): Image size
    Returns:
        np.ndarray: Rendered RGB image
    """
    extrinsic = gym.get_pose()   # 4x4 world->camera
    img = renderer.render_image_from_cam_param(K, extrinsic, width, height)
    return img


def interactive_control(ply_file="scans/scene0030_02/scene0030_02_vh_clean.ply", 
                        sens_file="scans/scene0030_02/scene0030_02.sens", 
                        frame_index=380,
                        width=400,
                        height=300,
                        is_discrete=True):
    """
    Interactive control loop:
    - w, a, s, d: translation
    - q, e: yaw rotation
    - r, f: pitch rotation
    - t, g: move up/down
    - save: save current view
    - quit: exit
    """
    
    # Load data
    sensor_data = SensorData(sens_file)
    renderer = Render(ply_file)
    gym = ViewManipulator(is_discrete=is_discrete)
    

    # Use ScanNet intrinsics (color camera intrinsics)
    K = sensor_data.intrinsic_color[:3, :3]
    target_frame = sensor_data.frames[frame_index]
    camera_to_world = target_frame['camera_to_world']
    gym.reset(camera_to_world)
    print(f"Current pose: {gym.get_se3()}")
    print("\n=== Interactive Camera Control ===")
    print("Commands: w/a/s/d (move), q/e (yaw), r/f (pitch), save (save image), quit (exit)")

    step_count = 0
    while True:
        img = render_current_view(renderer, gym, K, width, height)
        filename = f"view_step_{step_count}.png"
        Image.fromarray(img).save(filename)
        print(f"Saved current view to {filename}")
        cmd = input(f"\nStep {step_count} > Enter command: ").strip().lower()

        if cmd == "quit":
            print("Exiting...")
            break
        elif cmd in ["w", "a", "s", "d", "q", "e", "r", "f", "t", "g", "y", "h"]:
            gym.step(cmd)
            img = render_current_view(renderer, gym, K, width, height)
            print(f"Executed action: {cmd}, rendered image shape: {img.shape}")
            print(f"Current pose: {gym.get_se3()}")
        else:
            print("Invalid command. Use w/a/s/d/q/e/r/f/save/quit.")

        step_count += 1


if __name__ == "__main__":
    fire.Fire(interactive_control)
