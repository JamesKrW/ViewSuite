from view_suite.envs.base.view_base_env import ViewBaseEnv
from view_suite.scannet.render.mesh_render import MeshRenderer
from typing import Dict, Any
import os
from view_suite.scannet.unified_renderer import (
    UnifiedRender,
    DEFAULT_CLIENT_MAX_INFLIGHT,
    DEFAULT_CLIENT_OPEN_TIMEOUT,
)
from typing import List
from abc import abstractmethod
from typing import Tuple

class GymScannetRenderEnv(ViewBaseEnv):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        render_backend = env_config.get("render_backend","local")
        scannet_root = env_config.get("scannet_root",None)
        client_url = env_config.get("client_url",None)
        client_url_file_path = env_config.get("client_url_file_path",None)
        if client_url is None and client_url_file_path is not None:
            with open(client_url_file_path,"r") as f:
                client_url = f.read().strip()
        client_origin = env_config.get("client_origin",None)
        scene_id = env_config.get("scene_id",None)
        client_open_timeout = env_config.get("client_open_timeout", DEFAULT_CLIENT_OPEN_TIMEOUT)
        self.render_size = env_config.get("render_size", (512, 512))
        client_max_inflight = env_config.get("client_max_inflight", DEFAULT_CLIENT_MAX_INFLIGHT)
        self.renderer = UnifiedRender(
            render_backend=render_backend,
            scannet_root=scannet_root,
            client_url=client_url,
            client_origin=client_origin,
            scene_id=scene_id,
            client_open_timeout=client_open_timeout,
            client_max_inflight=client_max_inflight,
        )

    async def render_image_from_cam_param(self, camera_intrinsics, camera_extrinsics, width=None, height=None):
        if width is None:
            width = self.render_size[0]
        if height is None:
            height = self.render_size[1]
        return await self.renderer.render_image_from_cam_param(camera_intrinsics, camera_extrinsics, width, height)
    
    
    async def render_tasks(self, tasks: List[Dict[str, Any]]):
        return await self.renderer.render_tasks(tasks)
    
    @abstractmethod
    async def close(self) -> None:
        ...
    
    @abstractmethod
    async def system_prompt(self) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
    
    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ...


   
                
        
