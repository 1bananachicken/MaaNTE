from typing import Any


_pending_params: dict[str, Any] | None = None


def handoff_navigation(params: dict[str, Any]) -> None:
    """暂存在线地图配置，供紧随其后的实时辅助任务接管。"""
    global _pending_params
    _pending_params = dict(params)


def consume_navigation_handoff() -> dict[str, Any] | None:
    """获取并清除待接管的在线地图配置。"""
    global _pending_params
    params = _pending_params
    _pending_params = None
    return params
