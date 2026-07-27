import threading
from typing import Any


_pending_params: dict[int, dict[str, Any]] = {}
_pending_params_lock = threading.Lock()


def handoff_navigation(target_task_id: int, params: dict[str, Any]) -> None:
    """按目标任务暂存在线地图配置。"""
    with _pending_params_lock:
        _pending_params[target_task_id] = dict(params)


def consume_navigation_handoff(task_id: int) -> dict[str, Any] | None:
    """获取并清除明确交给当前任务的在线地图配置。"""
    with _pending_params_lock:
        return _pending_params.pop(task_id, None)
