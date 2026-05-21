from abc import ABC, abstractmethod
import numpy as np

class BaseRenderer(ABC):
    """Base class for both Mesh and PointCloud renderers."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.renderer = None
        # load 3d file here if needed

    @abstractmethod
    def render_image_from_cam_param(
        self,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        width: int = 300,
        height: int = 300,
    ):
        """
        Render an image from given camera parameters.

        Args:
            camera_intrinsics (np.ndarray):
                - Shape: (3, 3)
                - OpenCV-style pinhole intrinsics:
                    [[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]]
                - fx, fy: focal lengths in pixels
                - cx, cy: principal point in pixels
                - Must correspond to target image resolution (width, height).

            camera_extrinsics (np.ndarray):
                - Shape: (4, 4)
                - Camera-to-world (c2w) transformation matrix:

            width (int):
                Output image width in pixels.

            height (int):
                Output image height in pixels.

        Returns:
            np.ndarray: Rendered RGB image as (height, width, 3), dtype=uint8.
        """
        ...
