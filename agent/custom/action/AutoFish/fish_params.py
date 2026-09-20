"""钓鱼自定义动作的共享参数解析。"""

from __future__ import annotations

import json
import math

FISH_CONTROL_DEFAULTS = {
    "safe_margin": 6.0,
    "center_band_ratio": 0.4,
    "prediction_ms": 140.0,
    "velocity_alpha": 0.5,
    "green_velocity_alpha": 0.5,
    "green_center_alpha": 0.85,
    "pulse_min_ms": 18.0,
    "pulse_max_ms": 36.0,
    "pulse_ms_per_px": 0.45,
    "width_change_threshold": 8.0,
    "width_confirm_frames": 2,
    "control_end_grace_ms": 300.0,
    "lost_timeout_ms": 120.0,
    "lost_abort_ms": 1500.0,
    "loop_interval_ms": 0.0,
}


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


def _float_param(params: dict, name: str) -> float:
    default = float(FISH_CONTROL_DEFAULTS[name])
    value = params.get(name, default)
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int_param(params: dict, name: str) -> int:
    default = int(FISH_CONTROL_DEFAULTS[name])
    value = params.get(name, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def load_fish_control_params(custom_action_param) -> dict:
    """逐项解析控条参数，无效字段回退为默认值。"""
    params = load_custom_action_params(custom_action_param)
    pulse_min_ms = max(0.0, _float_param(params, "pulse_min_ms"))
    return {
        "safe_margin": max(0.0, _float_param(params, "safe_margin")),
        "center_band_ratio": max(
            0.0, min(1.0, _float_param(params, "center_band_ratio"))
        ),
        "prediction_ms": max(0.0, _float_param(params, "prediction_ms")),
        "velocity_alpha": _float_param(params, "velocity_alpha"),
        "green_velocity_alpha": _float_param(params, "green_velocity_alpha"),
        "green_center_alpha": max(
            0.0, min(1.0, _float_param(params, "green_center_alpha"))
        ),
        "pulse_min_ms": pulse_min_ms,
        "pulse_max_ms": max(pulse_min_ms, _float_param(params, "pulse_max_ms")),
        "pulse_ms_per_px": max(0.0, _float_param(params, "pulse_ms_per_px")),
        "width_change_threshold": max(
            0.1, _float_param(params, "width_change_threshold")
        ),
        "width_confirm_frames": max(1, _int_param(params, "width_confirm_frames")),
        "control_end_grace_ms": max(0.0, _float_param(params, "control_end_grace_ms")),
        "lost_timeout_ms": max(0.0, _float_param(params, "lost_timeout_ms")),
        "lost_abort_ms": max(0.0, _float_param(params, "lost_abort_ms")),
        "loop_interval_ms": max(0.0, _float_param(params, "loop_interval_ms")),
    }
