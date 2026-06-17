from __future__ import annotations

import json
import sys

from maa.custom_action import CustomAction
from maa.context import Context


DEFAULT_AUTO_RESIZE_GAME_WINDOW = True
AUTO_RESIZE_CONFIG_NODE = "PinkPawHeist_AutoResizeGameWindowConfig"

if sys.platform == "win32":
    try:
        try:
            from agent.utils.win32_process import ensure_game_window_resolution
        except ImportError:
            from utils.win32_process import ensure_game_window_resolution
    except ImportError:
        ensure_game_window_resolution = None
else:
    ensure_game_window_resolution = None


def _parse_custom_action_param(
    argv: CustomAction.RunArg,
    log_prefix="[PinkPawHeist]",
) -> dict:
    value = getattr(argv, "custom_action_param", None)
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception as exc:
        print(f"{log_prefix} invalid custom_action_param: {value!r}, error: {exc}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
    return bool(default)


def _get_auto_resize_game_window(
    ctx: Context,
    default=DEFAULT_AUTO_RESIZE_GAME_WINDOW,
) -> bool:
    try:
        node_data = ctx.get_node_data(AUTO_RESIZE_CONFIG_NODE) or {}
    except Exception:
        return bool(default)
    attach = node_data.get("attach") if isinstance(node_data, dict) else None
    if isinstance(attach, dict) and "auto_resize_game_window" in attach:
        return _parse_bool(attach.get("auto_resize_game_window"), default)
    return bool(default)
