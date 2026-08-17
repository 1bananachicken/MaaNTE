"""钓鱼自定义动作的共享参数解析。"""

from __future__ import annotations

import json


def load_custom_action_params(custom_action_param) -> dict:
    """将 CustomAction 参数统一解析为字典。"""
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    try:
        params = json.loads(custom_action_param)
    except (TypeError, ValueError):
        return {}
    return params if isinstance(params, dict) else {}
