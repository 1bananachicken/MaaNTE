from __future__ import annotations
from collections.abc import Sequence

BASELINE_WIDTH = 1280
BASELINE_HEIGHT = 720

_current_width = BASELINE_WIDTH
_current_height = BASELINE_HEIGHT
_scale_x = 1.0
_scale_y = 1.0


def current_size() -> tuple[int, int]:
    return _current_width, _current_height


def scaling_factors() -> tuple[float, float]:
    return _scale_x, _scale_y


def uniform_scale() -> float:
    return min(_scale_x, _scale_y)


def update_screen_size(width: int, height: int) -> None:
    """更新当前屏幕/游戏窗口尺寸并计算与基准分辨率的缩放系数。"""
    global _current_width, _current_height, _scale_x, _scale_y
    _current_width = int(width)
    _current_height = int(height)
    _scale_x = _current_width / BASELINE_WIDTH if BASELINE_WIDTH else 1.0
    _scale_y = _current_height / BASELINE_HEIGHT if BASELINE_HEIGHT else 1.0


def map_point(x: int, y: int) -> Sequence[int]:
    return int(round(x * _scale_x)), int(round(y * _scale_y))


def map_rect(rect: Sequence[int]) -> Sequence[int]:
    x, y, w, h = rect
    mapped_x, mapped_y = map_point(x, y)
    return mapped_x, mapped_y, int(round(w * _scale_x)), int(round(h * _scale_y))
