from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

BASELINE_WIDTH = 1280
BASELINE_HEIGHT = 720
BASELINE_ASPECT_RATIO = BASELINE_WIDTH / BASELINE_HEIGHT
EDGE_ANCHOR_LEFT_TOP_RATIO = 0.35
EDGE_ANCHOR_RIGHT_BOTTOM_RATIO = 0.65
SCALE_KEY_PRECISION = 1000

_current_width = BASELINE_WIDTH
_current_height = BASELINE_HEIGHT
_scale_x = 1.0
_scale_y = 1.0

_frame_width = BASELINE_WIDTH
_frame_height = BASELINE_HEIGHT
_frame_scale_x = 1.0
_frame_scale_y = 1.0


class RectMapping(NamedTuple):
    name: str
    rect: tuple[int, int, int, int]
    scale_x: float
    scale_y: float


def _calc_scale(width: int, height: int) -> tuple[float, float]:
    return width / BASELINE_WIDTH, height / BASELINE_HEIGHT


def _has_size(width: int, height: int) -> bool:
    return width > 0 and height > 0


def current_size() -> tuple[int, int]:
    """Return the currently detected game client/input size."""
    return _current_width, _current_height


def frame_size() -> tuple[int, int]:
    """Return the latest screenshot frame size."""
    return _frame_width, _frame_height


def scaling_factors() -> tuple[float, float]:
    """Return scale factors from 1280x720 baseline to the current input size."""
    return _scale_x, _scale_y


def frame_scaling_factors() -> tuple[float, float]:
    """Return scale factors from 1280x720 baseline to the screenshot frame."""
    return _frame_scale_x, _frame_scale_y


def uniform_scale() -> float:
    """Return a conservative uniform scale for input size-like values."""
    return min(_scale_x, _scale_y)


def frame_uniform_scale() -> float:
    """Return a conservative uniform scale for screenshot size-like values."""
    return min(_frame_scale_x, _frame_scale_y)


def is_baseline_size(width: int | None = None, height: int | None = None) -> bool:
    width = _current_width if width is None else width
    height = _current_height if height is None else height
    return width == BASELINE_WIDTH and height == BASELINE_HEIGHT


def is_supported_16_9(
    width: int | None = None,
    height: int | None = None,
    *,
    tolerance: float = 0.01,
) -> bool:
    width = _current_width if width is None else width
    height = _current_height if height is None else height
    if not _has_size(width, height):
        return False

    return abs((width / height) - BASELINE_ASPECT_RATIO) <= tolerance


def update_screen_size(width: int, height: int) -> None:
    """Update global input scale used by custom actions."""
    global _current_width, _current_height, _scale_x, _scale_y

    if not _has_size(width, height):
        return

    _current_width = int(width)
    _current_height = int(height)
    _scale_x, _scale_y = _calc_scale(_current_width, _current_height)


def update_frame_size(width: int, height: int) -> None:
    """Update global screenshot scale used by image matching."""
    global _frame_width, _frame_height, _frame_scale_x, _frame_scale_y

    if not _has_size(width, height):
        return

    _frame_width = int(width)
    _frame_height = int(height)
    _frame_scale_x, _frame_scale_y = _calc_scale(_frame_width, _frame_height)


def update_frame_size_from_image(image: Any) -> None:
    """Update screenshot scale from a numpy-like image object."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        return

    height, width = shape[:2]
    update_frame_size(int(width), int(height))


def _map_point(
    x: int | float,
    y: int | float,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int]:
    return round(x * scale_x), round(y * scale_y)


def _map_point_offset(
    x: int | float,
    y: int | float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int]:
    return round(offset_x + x * scale_x), round(offset_y + y * scale_y)


def _map_rect(
    rect: Sequence[int | float],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    if len(rect) < 4:
        raise ValueError("rect must contain x, y, w, h")

    x, y, w, h = rect[:4]
    x1, y1 = _map_point(x, y, scale_x, scale_y)
    x2, y2 = _map_point(x + w, y + h, scale_x, scale_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _map_rect_offset(
    rect: Sequence[int | float],
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int, int, int]:
    if len(rect) < 4:
        raise ValueError("rect must contain x, y, w, h")

    x, y, w, h = rect[:4]
    x1, y1 = _map_point_offset(x, y, scale_x, scale_y, offset_x, offset_y)
    x2, y2 = _map_point_offset(x + w, y + h, scale_x, scale_y, offset_x, offset_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _fit_transform(width: int, height: int) -> tuple[float, float, float, float]:
    scale = min(width / BASELINE_WIDTH, height / BASELINE_HEIGHT)
    offset_x = (width - BASELINE_WIDTH * scale) / 2
    offset_y = (height - BASELINE_HEIGHT * scale) / 2
    return scale, scale, offset_x, offset_y


def _fill_transform(width: int, height: int) -> tuple[float, float, float, float]:
    scale = max(width / BASELINE_WIDTH, height / BASELINE_HEIGHT)
    offset_x = (width - BASELINE_WIDTH * scale) / 2
    offset_y = (height - BASELINE_HEIGHT * scale) / 2
    return scale, scale, offset_x, offset_y


def _edge_aware_rect(
    rect: Sequence[int | float],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    """Map HUD-like rects by anchoring near-edge regions to their nearest edge."""
    if len(rect) < 4:
        raise ValueError("rect must contain x, y, w, h")

    x, y, w, h = rect[:4]
    center_x = x + w / 2
    center_y = y + h / 2

    if center_x < BASELINE_WIDTH * EDGE_ANCHOR_LEFT_TOP_RATIO:
        mapped_x = round(x * scale)
    elif center_x > BASELINE_WIDTH * EDGE_ANCHOR_RIGHT_BOTTOM_RATIO:
        mapped_x = round(width - (BASELINE_WIDTH - x) * scale)
    else:
        mapped_x = round(width / 2 + (center_x - BASELINE_WIDTH / 2) * scale - w * scale / 2)

    if center_y < BASELINE_HEIGHT * EDGE_ANCHOR_LEFT_TOP_RATIO:
        mapped_y = round(y * scale)
    elif center_y > BASELINE_HEIGHT * EDGE_ANCHOR_RIGHT_BOTTOM_RATIO:
        mapped_y = round(height - (BASELINE_HEIGHT - y) * scale)
    else:
        mapped_y = round(height / 2 + (center_y - BASELINE_HEIGHT / 2) * scale - h * scale / 2)

    return mapped_x, mapped_y, max(1, round(w * scale)), max(1, round(h * scale))


def map_point(x: int | float, y: int | float) -> tuple[int, int]:
    """Map a 1280x720 point to the current input size."""
    return map_point_to_input(x, y)


def map_rect(rect: Sequence[int | float]) -> tuple[int, int, int, int]:
    """Map a 1280x720 rect (x, y, w, h) to the current input size."""
    return map_rect_to_input(rect)


def map_point_to_input(x: int | float, y: int | float) -> tuple[int, int]:
    return _map_point(x, y, _scale_x, _scale_y)


def map_rect_to_input(rect: Sequence[int | float]) -> tuple[int, int, int, int]:
    return _map_rect(rect, _scale_x, _scale_y)


def map_point_to_frame(x: int | float, y: int | float) -> tuple[int, int]:
    return _map_point(x, y, _frame_scale_x, _frame_scale_y)


def map_rect_to_frame(rect: Sequence[int | float]) -> tuple[int, int, int, int]:
    return _map_rect(rect, _frame_scale_x, _frame_scale_y)


def map_rect_to_frame_fit(rect: Sequence[int | float]) -> tuple[int, int, int, int]:
    scale_x, scale_y, offset_x, offset_y = _fit_transform(_frame_width, _frame_height)
    return _map_rect_offset(rect, scale_x, scale_y, offset_x, offset_y)


def map_rect_to_frame_candidates(rect: Sequence[int | float]) -> tuple[RectMapping, ...]:
    """Return plausible frame-space mappings for a 1280x720 baseline rect.

    The first candidate preserves the normal Maa-style stretch mapping. Non-16:9
    frames add fit/fill and edge-anchored variants to cover common game layouts.
    """
    candidates: list[RectMapping] = [
        RectMapping("stretch", map_rect_to_frame(rect), _frame_scale_x, _frame_scale_y)
    ]

    if not is_supported_16_9(_frame_width, _frame_height):
        fit_scale_x, fit_scale_y, fit_offset_x, fit_offset_y = _fit_transform(_frame_width, _frame_height)
        candidates.append(
            RectMapping(
                "fit",
                _map_rect_offset(rect, fit_scale_x, fit_scale_y, fit_offset_x, fit_offset_y),
                fit_scale_x,
                fit_scale_y,
            )
        )

        fill_scale_x, fill_scale_y, fill_offset_x, fill_offset_y = _fill_transform(_frame_width, _frame_height)
        candidates.append(
            RectMapping(
                "fill",
                _map_rect_offset(rect, fill_scale_x, fill_scale_y, fill_offset_x, fill_offset_y),
                fill_scale_x,
                fill_scale_y,
            )
        )

        height_scale = _frame_height / BASELINE_HEIGHT
        candidates.append(
            RectMapping(
                "edge_height",
                _edge_aware_rect(rect, _frame_width, _frame_height, height_scale),
                height_scale,
                height_scale,
            )
        )

        width_scale = _frame_width / BASELINE_WIDTH
        candidates.append(
            RectMapping(
                "edge_width",
                _edge_aware_rect(rect, _frame_width, _frame_height, width_scale),
                width_scale,
                width_scale,
            )
        )

    unique: list[RectMapping] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for candidate in candidates:
        clamped = clamp_rect(candidate.rect, _frame_width, _frame_height)
        key = (
            clamped[0],
            clamped[1],
            clamped[2],
            clamped[3],
            round(candidate.scale_x * SCALE_KEY_PRECISION),
            round(candidate.scale_y * SCALE_KEY_PRECISION),
        )
        if clamped[2] <= 0 or clamped[3] <= 0 or key in seen:
            continue
        seen.add(key)
        unique.append(RectMapping(candidate.name, clamped, candidate.scale_x, candidate.scale_y))

    return tuple(unique)


def clamp_rect(
    rect: Sequence[int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(rect) < 4:
        raise ValueError("rect must contain x, y, w, h")

    x, y, w, h = rect[:4]
    x = max(0, min(int(x), width))
    y = max(0, min(int(y), height))
    w = max(0, min(int(w), width - x))
    h = max(0, min(int(h), height - y))
    return x, y, w, h


def resize_template(
    template: Any,
    scale_x: float,
    scale_y: float,
    *,
    green_mask: bool = False,
) -> Any:
    """Scale a baseline template by explicit x/y scale factors."""
    if template is None:
        return template

    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return template

    shape = getattr(template, "shape", None)
    if shape is None or len(shape) < 2:
        return template

    height, width = shape[:2]
    scaled_width = max(1, round(width * scale_x))
    scaled_height = max(1, round(height * scale_y))
    if scaled_width == width and scaled_height == height:
        return template

    import cv2

    if green_mask:
        interpolation = cv2.INTER_NEAREST
    elif scale_x < 1.0 or scale_y < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(template, (scaled_width, scaled_height), interpolation=interpolation)


def resize_template_to_frame(template: Any, *, green_mask: bool = False) -> Any:
    """Scale a 1280x720-baseline template to the latest screenshot frame."""
    return resize_template(template, _frame_scale_x, _frame_scale_y, green_mask=green_mask)
