import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv
from PIL import Image


from view_suite.envs.sokoban.utils.env_config import SokobanEnvConfig
from view_suite.envs.sokoban.utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from view_suite.envs.sokoban.utils.utils import parse_free_think, numpy_to_pil


from view_suite.gym.gym_image_env import GymImageEnv


import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
from PIL import Image



class Sokoban(GymImageEnv):
    """
    Sokoban environment that implements the EnvImageBase async interface.
    Uses asyncio.to_thread(...) to offload blocking gym calls (reset/step/render/close)
    to a thread pool so the event loop is not blocked.
    """

    # Text rendering lookup
    GRID_LOOKUP = {
        0: " # ",  # wall
        1: " _ ",  # floor
        2: " O ",  # target
        3: " √ ",  # box on target
        4: " X ",  # box
        5: " P ",  # player
        6: " S ",  # player on target
    }

    # Action mapping
    ACTION_LOOKUP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
    }

    def __init__(self, env_config: Dict[str, Any]):
        """
        env_config:
        dim_room: tuple = (6, 6)  # Room dimensions (height, width)
        max_steps: int = 100      # Maximum steps per episode
        num_boxes: int = 1        # Number of boxes in the room
        render_mode: str = "text" # "text" or "vision"
        max_actions_per_step: int = 3  # Max actions per step
        action_sep: str = ","     # Separator between actions
        image_placeholder: str = "<image>"  # Placeholder for vision mode
        """
        super().__init__(env_config)
        self.env_config = env_config
        self.config = SokobanEnvConfig(**self.env_config)
        # Create the underlying (blocking) gym env
        self.env = SokobanEnv(
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
        )
        self.total_reward: float = 0.0
        self.valid_actions: List[str] = []

    # ------------------------------
    # EnvImageBase abstract methods
    # ------------------------------
    async def close(self) -> None:
        """Non-blocking close via to_thread to avoid blocking the loop."""
        await asyncio.to_thread(self.env.close)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Non-blocking reset:
        - Offloads env.reset() to a thread pool to avoid blocking the event loop.
        """
        # If seeding is needed, set it before reset in to_thread, or call a seeded reset API.
        await asyncio.to_thread(self.env.reset)
        self.total_reward = 0.0
        self.valid_actions = []
        obs = await self._render_async(init_obs=True)
        info: Dict[str, Any] = {}
        return obs, info

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Non-blocking system prompt:
        - Offloads system prompt to a thread pool to avoid blocking the event loop.
        """
        
        return {
            "obs_str": self.get_system_prompt(),
        }

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Non-blocking step:
        - Parses action_str
        - Offloads env.step(...) to thread pool for each primitive action
        - Computes metrics, reward shaping, success, etc.
        """
        parsed = parse_free_think(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        self.valid_actions = []
        info.update(parsed)
        action_list: List[str] = parsed.get("actions", [])
        # Copy current player position (read-only)
        prev_player_pos = np.array(self.env.player_position, copy=True)

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and parsed.get("format_correct", False),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int = self.ACTION_LOOKUP[action]
                # Offload the blocking gym step to a thread
                _obs, step_reward, step_done, _ = await asyncio.to_thread(self.env.step, action_int)
                reward += float(step_reward)
                self.valid_actions.append(action)
                # Early success check
                if self._is_success():
                    done = True
                    metrics["traj_metrics"]["success"] = True
                    break
            else:
                metrics["turn_metrics"]["action_is_valid"] = False
                break

        # Keep your shaping logic (no-op here)
        if self.valid_actions:
            reward += 0.0

        # Effective action: detect player position change
        metrics["turn_metrics"]["action_is_effective"] = not np.array_equal(
            prev_player_pos, self.env.player_position
        )

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        obs = await self._render_async(init_obs=False)
        return obs, reward, done, info

    # ------------------------------
    # Public helpers
    # ------------------------------
    def get_system_prompt(self) -> str:
        """Keep original system prompt composition."""
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
        )
        return system_prompt() + "\n" + format_prompt_str

    # ------------------------------
    # Internal helpers
    # ------------------------------
    async def _render_async(self, init_obs: bool) -> Dict[str, Any]:
        """
        Async wrapper of render to avoid blocking:
        - For vision mode, offloads env.render(mode="rgb_array") to thread pool.
        - For text mode, uses current room_state to format grid text.
        """
        multi_modal_input: Optional[Dict[str, List[Image.Image]]] = None

        # Build format prompt (without example in obs)
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
        )

        if self.config.render_mode == "vision":
            # Offload blocking render to a thread pool
            rgb_array = await asyncio.to_thread(self.env.render, "rgb_array")
            img_str = self.config.image_placeholder
            multi_modal_input = {
                self.config.image_placeholder: [numpy_to_pil(rgb_array)]
            }
        else:
            img_str = self._grid_to_text()

        if init_obs:
            obs_str = init_observation_template(img_str) + "\n" #+ format_prompt_str
        else:
            obs_str = action_template(self.valid_actions, img_str) + "\n" #+ format_prompt_str

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input
        return obs

    def _grid_to_text(self) -> str:
        """Convert current room_state to a human-readable text grid."""
        room_state = np.where(
            (self.env.room_state == 5) & (self.env.room_fixed == 2),
            6,
            self.env.room_state,
        )
        text_rows = []
        for row in room_state:
            text_row = "".join(self.GRID_LOOKUP.get(int(cell), "?") for cell in row)
            text_rows.append(text_row)
        return "\n".join(text_rows)

    def _is_success(self) -> bool:
        """Check if all boxes are on targets."""
        return self.env.boxes_on_target == self.env.num_boxes


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import os

    async def main_async(render_mode: str = "vision",
                         num_boxes: int = 1,
                         dim_room: Tuple[int, int] = (5, 5),
                         max_actions_per_step: int = 2,
                         save_path: str = "./test"):
        cfg = {
            "render_mode": render_mode,
            "num_boxes": num_boxes,
            "dim_room": dim_room,
            "max_actions_per_step": max_actions_per_step,
        }
        env = Sokoban(cfg)

        print("System Prompt:")
        print(env.get_system_prompt())
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print("Initial Observation:")
        print(obs["obs_str"])
        step=0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            # save the image to target folder
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))
        step+=1
        while True:
            print(f"\nStep {step + 1}:")
            try:
                action_input = input("Enter action string (or 'quit'): ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            if not action_input.startswith("<think>"):
                action_input = f"<think>Moving towards the goal.</think><answer>{action_input}</answer>"

            obs, reward, done, info = await env.step(action_input)
            if "multi_modal_input" in obs:
                # save the image to target folder
                img = obs["multi_modal_input"][env.config.image_placeholder][0]
                img.save(os.path.join(save_path, f"step_{step}.png"))
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation:\n{obs['obs_str']}")
            if done:
                print("Puzzle solved!")
                break
            step+=1

        print(f"\nTotal reward: {env.total_reward}")
        await env.close()

    def main(render_mode: str = "vision",
             num_boxes: int = 1,
             dim_room: Tuple[int, int] = (5, 5),
             max_actions_per_step: int = 2,
             save_path: str = "./test"):
        asyncio.run(main_async(
            render_mode=render_mode,
            num_boxes=num_boxes,
            dim_room=dim_room,
            max_actions_per_step=max_actions_per_step,
            save_path=save_path
        ))

    fire.Fire(main)
