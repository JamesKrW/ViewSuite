# -*- coding: utf-8 -*-
# Format checking utilities for GymImageEnv-style async envs.
# Drop-in: no subclass signature change needed.

# Usage:
# env = MyGymImageEnv(env_config)
# auto_wrap_env_methods(env, default_enabled=False)  
# obs, reward, done, info = await env.step("do something", format_checking=True) # one-time
# env.enable_format_check(True) # enable by default
# obs, info = await env.reset(seed=42) 



from __future__ import annotations
import inspect
import logging
import numbers
import re
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("EnvFormatCheck")
if not logger.handlers:
    # Minimal default handler; integrate with your logging config as needed.
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.WARNING)


# ---------- Low-level validators ----------

_IMAGE_TOKEN_PATTERN = re.compile(r"<image>")

def _count_image_tokens(text: str) -> int:
    """Count <image> placeholders in obs_str."""
    return len(_IMAGE_TOKEN_PATTERN.findall(text or ""))

def validate_obs(obs: Dict[str, Any]) -> None:
    """Validate observation dict structure and image placeholder alignment."""
    if not isinstance(obs, dict):
        raise AssertionError(f"obs must be dict, got {type(obs)}")

    if "obs_str" not in obs:
        raise AssertionError("obs must contain key 'obs_str'")
    if not isinstance(obs["obs_str"], str):
        raise AssertionError(f"obs['obs_str'] must be str, got {type(obs['obs_str'])}")

    n_tokens = _count_image_tokens(obs["obs_str"])
    mmi = obs.get("multi_modal_input", None)

    img_list = None
    if mmi is not None:
        if not isinstance(mmi, dict):
            raise AssertionError(f"obs['multi_modal_input'] must be dict, got {type(mmi)}")
        if "<image>" in mmi:
            img_list = mmi["<image>"]
            if not isinstance(img_list, list):
                raise AssertionError(
                    f"obs['multi_modal_input']['<image>'] must be a list, got {type(img_list)}"
                )
            for i, im in enumerate(img_list):
                if not isinstance(im, Image.Image):
                    raise AssertionError(
                        f"multi_modal_input['<image>'][{i}] must be PIL.Image.Image, got {type(im)}"
                    )

    # Bidirectional consistency checks between obs_str and image list.
    if n_tokens > 0 and (img_list is None or len(img_list) == 0):
        raise AssertionError(
            f"obs_str contains {n_tokens} '<image>' placeholders but "
            f"multi_modal_input['<image>'] is missing or empty."
        )
    if img_list is not None and len(img_list) > 0 and n_tokens == 0:
        raise AssertionError(
            "obs has multi_modal_input['<image>'] images but obs_str has no '<image>' placeholders."
        )
    if img_list is not None and n_tokens > 0 and len(img_list) != n_tokens:
        raise AssertionError(
            f"Number of '<image>' placeholders ({n_tokens}) does not match "
            f"number of images ({len(img_list)})."
        )

def validate_info(info: Dict[str, Any]) -> None:
    """Validate info dict shape and optional 'success' semantics."""
    if not isinstance(info, dict):
        raise AssertionError(f"info must be dict, got {type(info)}")

    if "success" not in info:
        logger.warning("`info` has no 'success' key; we require it to evaluate episode correctness.")
    else:
        if not isinstance(info["success"], bool):
            raise AssertionError(f"info['success'] must be bool, got {type(info['success'])}")

def validate_reward(reward: Any) -> None:
    """Validate reward is numeric (float-like)."""
    # Accept any real number (int/float/numpy scalar); reject bool.
    if isinstance(reward, bool) or not isinstance(reward, numbers.Real):
        raise AssertionError(f"reward must be a real number (float-like), got {type(reward)}")

def validate_done(done: Any) -> None:
    """Validate done is boolean."""
    if not isinstance(done, bool):
        raise AssertionError(f"done must be bool, got {type(done)}")


# ---------- Decorator (can be used on subclass methods directly) ----------

def format_check(kind: str):
    """
    Decorator for async env methods:
      - kind='system_prompt' -> expects: obs
      - kind='reset'         -> expects: (obs, info)
      - kind='step'          -> expects: (obs, reward, done, info)

    Usage (optional):
      @format_check('step')
      async def step(self, action_str: str):
          ...
    """
    if kind not in {"system_prompt", "reset", "step"}:
        raise ValueError(f"Unsupported kind={kind}")

    def _decorator(fn):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("@format_check must wrap an async def")

        @wraps(fn)
        async def _wrapped(self, *args, **kwargs):
            # Allow per-call toggle without leaking unknown kwargs to original fn.
            check_flag = kwargs.pop("format_checking", None)
            if check_flag is None:
                check_flag = bool(getattr(self, "_format_check_default", False))

            result = await fn(self, *args, **kwargs)

            if check_flag:
                if kind == "system_prompt":
                    validate_obs(result)
                elif kind == "reset":
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 2
                    ):
                        raise AssertionError("reset must return (obs, info)")
                    obs, info = result
                    validate_obs(obs)
                    validate_info(info)
                elif kind == "step":
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 4
                    ):
                        raise AssertionError("step must return (obs, reward, done, info)")
                    obs, reward, done, info = result
                    validate_obs(obs)
                    validate_reward(reward)
                    validate_done(done)
                    validate_info(info)
            return result

        # Mark to avoid double-wrapping later.
        setattr(_wrapped, "_is_format_wrapped", True)
        return _wrapped

    return _decorator


# ---------- Instance-level auto-wrapping (no subclass changes required) ----------

def auto_wrap_env_methods(env: Any, default_enabled: bool = False) -> Any:
    """
    Monkey-patch an env instance so that:
      - env.system_prompt(..., format_checking=True) works
      - env.reset(..., format_checking=True) works
      - env.step(..., format_checking=True) works
    And you can flip default via env.enable_format_check(True/False).

    This does NOT change the underlying subclass method signatures.
    """
    setattr(env, "_format_check_default", bool(default_enabled))

    def _wrap_bound(name: str, kind: str):
        bound = getattr(env, name, None)
        if bound is None:
            return
        # If already wrapped, skip
        if getattr(bound, "_is_format_wrapped", False):
            return
        if not inspect.iscoroutinefunction(bound):
            raise TypeError(f"env.{name} must be async def to auto-wrap")

        # Get original function (unbound) then bind wrapper back to the instance.
        fn = getattr(bound, "__func__", None) or bound
        wrapped = format_check(kind)(fn)
        setattr(env, name, MethodType(wrapped, env))

    _wrap_bound("system_prompt", "system_prompt")
    _wrap_bound("reset", "reset")
    _wrap_bound("step", "step")

    # Add a small helper to toggle default checking.
    def enable_format_check(self, enabled: bool = True):
        """Turn on/off default format checking for this instance."""
        setattr(self, "_format_check_default", bool(enabled))
        return self

    # Attach helper only once.
    if not hasattr(env, "enable_format_check"):
        setattr(env, "enable_format_check", MethodType(enable_format_check, env))

    return env
