from view_suite.gym.gym_base_env import GymBaseEnv

from typing import Dict, Any, Tuple
from abc import abstractmethod
from typing import Optional, Union, List
from PIL import Image
class GymImageEnv(GymBaseEnv):
    """
    Same API as EnvBase, but if the env wants to return images, it MUST embed them
    into obs at:
        obs["multi_modal_input"]["<image>"] = [PIL.Image.Image, ...]
        
    Also, we should have <image> as placeholder in obs_str, and the number of <image>
    placeholders should match the number of images in the list above.
    please see format_utils.validate_obs for the exact validation logic.
    The handler will extract, serialize, and strip them from JSON.
    """
    
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def system_prompt(self) -> Dict[str, Any]:
        """
        Required obs shape when images are present:

            obs = {
              "obs_str": "...",
              "multi_modal_input": {
                  "<image>": [PIL.Image.Image, ...]
              }
            }
        """
        raise NotImplementedError
    
    @abstractmethod
    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Required obs shape when images are present:

            obs = {
              "obs_str": "...",
              "multi_modal_input": {
                  "<image>": [PIL.Image.Image, ...]
              }
            }
        info is a dict
        """
        raise NotImplementedError

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Required obs shape when images are present:

            obs = {
              "obs_str": "...",
              "multi_modal_input": {
                  "<image>": [PIL.Image.Image, ...]
              }
            }

        reward is a float
        done is a bool
        info is a dict
        """
        raise NotImplementedError