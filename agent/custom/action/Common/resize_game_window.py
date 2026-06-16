import json
import sys

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

if sys.platform == "win32":
    try:
        from utils.win32_process import ensure_game_window_resolution
    except Exception:
        ensure_game_window_resolution = None
else:
    ensure_game_window_resolution = None


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
    return bool(default)


def _parse_resize_params(raw_param):
    if not raw_param:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT

    if isinstance(raw_param, dict):
        params = raw_param
    else:
        try:
            params = json.loads(raw_param)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid custom_action_param: {exc}") from exc

    if not isinstance(params, dict):
        raise ValueError("custom_action_param must be an object")

    resolution = params.get("resolution")
    if resolution is not None:
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("resolution must be [width, height]")
        width, height = resolution
    else:
        width = params.get("width", DEFAULT_WIDTH)
        height = params.get("height", DEFAULT_HEIGHT)

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    return width, height


def _parse_optional_resize_kwargs(raw_param):
    if not raw_param:
        return {}
    if isinstance(raw_param, dict):
        params = raw_param
    else:
        params = json.loads(raw_param)
    if not isinstance(params, dict):
        return {}

    kwargs = {}
    if "process_name" in params:
        kwargs["process_name"] = str(params["process_name"])
    if "center" in params:
        kwargs["center"] = _parse_bool(params["center"], True)
    if "tolerance" in params:
        kwargs["tolerance"] = int(params["tolerance"])
    if "settle_ms" in params:
        kwargs["settle_ms"] = int(params["settle_ms"])
    return kwargs


@AgentServer.custom_action("resize_game_window")
class ResizeGameWindow(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        try:
            width, height = _parse_resize_params(argv.custom_action_param)
            kwargs = _parse_optional_resize_kwargs(argv.custom_action_param)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("resize_game_window 参数解析失败: %s", exc)
            return CustomAction.RunResult(success=False)

        if ensure_game_window_resolution is None:
            logger.warning("resize_game_window 仅支持 Windows 或 win32_process 不可用")
            return CustomAction.RunResult(success=False)

        try:
            result = ensure_game_window_resolution(width, height, **kwargs)
        except Exception as exc:
            logger.exception("resize_game_window 执行失败: %s", exc)
            return CustomAction.RunResult(success=False)

        success = bool(result.get("success")) if isinstance(result, dict) else bool(result)
        if success:
            logger.debug("resize_game_window 成功: %s", result)
        else:
            logger.warning("resize_game_window 失败: %s", result)
        return CustomAction.RunResult(success=success)
