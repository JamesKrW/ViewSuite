from view_suite.gym.gym_image_env import GymImageEnv

from typing import Dict, Any, Tuple
from abc import abstractmethod
from typing import Optional, Union, List
from PIL import Image
class ViewBaseEnv(GymImageEnv):
    """
    Same API as EnvBase, but if the env wants to return images, it MUST embed them
    into obs at:
        obs["multi_modal_input"]["<image>"] = [PIL.Image.Image, ...]
        
    Also, we should have <image> as placeholder in obs_str, and the number of <image>
    placeholders should match the number of images in the list above.
    please see format_utils.validate_obs for the exact validation logic.
    The handler will extract, serialize, and strip them from JSON.
    """

    def _resize_image(self, imgs: List[Image.Image]) -> List[Image.Image]:
        """
        If self.resize is set, return resized copies; otherwise return originals.
        Uses LANCZOS for high-quality down/up-sampling.
        """
        image_size=getattr(self, "image_size", None)
        if image_size is None:
            return imgs
        w= image_size[0]
        h= image_size[1]
        out: List[Image.Image] = []
        for im in imgs:
            if im.size == (w, h):
                out.append(im)
            else:
                out.append(im.resize((w, h), resample=Image.Resampling.LANCZOS))
        return out
    
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        

    