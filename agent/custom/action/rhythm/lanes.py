"""将配置中的比例坐标转换为像素 ROI。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LaneLayout:
    frame_w: int
    frame_h: int
    center_x: list[int]
    top_center_x: list[int]
    bottom_center_x: list[int]
    judge_center_y: list[int]
    half_width_px: int
    judge_y0: int
    judge_y1: int
    judge_y0_by_lane: list[int]
    judge_y1_by_lane: list[int]
    roi_y0: int
    roi_y1: int


def _frac(cfg: dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    return float(v)


def build_lane_layout(cfg: dict[str, Any], frame_w: int, frame_h: int) -> LaneLayout:
    lanes = cfg.get("lanes") or {}
    centers = list(lanes.get("center_x_frac") or [0.36, 0.44, 0.56, 0.64])
    if len(centers) != 4:
        raise ValueError("lanes.center_x_frac 必须为 4 个数")
    top_centers = list(lanes.get("top_center_x_frac") or centers)
    if len(top_centers) != 4:
        raise ValueError("lanes.top_center_x_frac 必须为 4 个数")

    half_w_frac = float(lanes.get("half_width_frac", 0.028))
    judge_y = _frac(lanes, "judge_line_y_frac", 0.82)
    judge_y_by_lane = list(lanes.get("judge_line_y_frac_by_lane") or [judge_y, judge_y, judge_y, judge_y])
    if len(judge_y_by_lane) != 4:
        raise ValueError("lanes.judge_line_y_frac_by_lane 必须为 4 个数")
    band_half = _frac(lanes, "judge_band_half_height_frac", 0.035)
    roi_top = _frac(lanes, "roi_top_y_frac", 0.35)

    cy = int(round(judge_y * frame_h))
    half_band = max(2, int(round(band_half * frame_h)))
    full_top = max(0, cy - half_band)
    full_bottom = min(frame_h, cy + half_band)
    band_px = full_bottom - full_top
    # 只保留判定带「靠上」的一段，避免条带伸到鼓面/光晕里导致黄轨 (F) 常亮误触
    keep_top = float(lanes.get("judge_band_keep_from_top", 1.0))
    keep_top = max(0.15, min(keep_top, 1.0))
    judge_y0 = full_top
    judge_y1 = min(frame_h, full_top + max(4, int(round(band_px * keep_top))))
    judge_center_y: list[int] = []
    judge_y0_by_lane: list[int] = []
    judge_y1_by_lane: list[int] = []
    for lane_judge_y in judge_y_by_lane:
        lcy = int(round(float(lane_judge_y) * frame_h))
        l_full_top = max(0, lcy - half_band)
        l_full_bottom = min(frame_h, lcy + half_band)
        l_band_px = l_full_bottom - l_full_top
        judge_center_y.append(lcy)
        judge_y0_by_lane.append(l_full_top)
        judge_y1_by_lane.append(min(frame_h, l_full_top + max(4, int(round(l_band_px * keep_top)))))

    roi_y0 = max(0, int(round(roi_top * frame_h)))
    roi_y1 = min(frame_h, judge_y1)

    half_width_px = max(2, int(round(half_w_frac * frame_w)))
    bottom_center_x = [int(round(c * frame_w)) for c in centers]
    top_center_x = [int(round(c * frame_w)) for c in top_centers]
    center_x = [
        lane_center_x_at_y_raw(top_center_x[i], bottom_center_x[i], roi_y0, judge_y1, (judge_y0 + judge_y1) // 2)
        for i in range(4)
    ]

    return LaneLayout(
        frame_w=frame_w,
        frame_h=frame_h,
        center_x=center_x,
        top_center_x=top_center_x,
        bottom_center_x=bottom_center_x,
        judge_center_y=judge_center_y,
        half_width_px=half_width_px,
        judge_y0=judge_y0,
        judge_y1=judge_y1,
        judge_y0_by_lane=judge_y0_by_lane,
        judge_y1_by_lane=judge_y1_by_lane,
        roi_y0=roi_y0,
        roi_y1=roi_y1,
    )


def lane_center_x_at_y_raw(top_x: int, bottom_x: int, top_y: int, bottom_y: int, y: int) -> int:
    if bottom_y <= top_y:
        return bottom_x
    t = (y - top_y) / float(bottom_y - top_y)
    t = max(0.0, min(1.0, t))
    return int(round(top_x + (bottom_x - top_x) * t))


def lane_center_x_at_y(layout: LaneLayout, lane_index: int, y: int) -> int:
    return lane_center_x_at_y_raw(
        layout.top_center_x[lane_index],
        layout.bottom_center_x[lane_index],
        layout.roi_y0,
        layout.judge_y1,
        y,
    )


def lane_roi_quad(layout: LaneLayout, lane_index: int, y0: int, y1: int) -> list[tuple[int, int]]:
    """返回沿斜轨道展开的四边形：左上、右上、右下、左下。"""
    cx0 = lane_center_x_at_y(layout, lane_index, y0)
    cx1 = lane_center_x_at_y(layout, lane_index, y1)
    hw = layout.half_width_px
    return [
        (max(0, cx0 - hw), y0),
        (min(layout.frame_w, cx0 + hw), y0),
        (min(layout.frame_w, cx1 + hw), y1),
        (max(0, cx1 - hw), y1),
    ]


def lane_judge_slice(layout: LaneLayout, lane_index: int) -> tuple[int, int, int, int]:
    """单轨在判定带内的 (x0, x1, y0, y1)，含 x 边界裁剪。"""
    y0 = layout.judge_y0_by_lane[lane_index]
    y1 = layout.judge_y1_by_lane[lane_index]
    cy = (y0 + y1) // 2
    cx = lane_center_x_at_y(layout, lane_index, cy)
    x0 = max(0, cx - layout.half_width_px)
    x1 = min(layout.frame_w, cx + layout.half_width_px)
    return x0, x1, y0, y1


def lane_full_roi_slice(layout: LaneLayout, lane_index: int) -> tuple[int, int, int, int]:
    """用于兼容旧调试逻辑的斜轨道外接矩形 ROI。"""
    quad = lane_roi_quad(layout, lane_index, layout.roi_y0, layout.roi_y1)
    xs = [p[0] for p in quad]
    x0 = max(0, min(xs))
    x1 = min(layout.frame_w, max(xs))
    return x0, x1, layout.roi_y0, layout.roi_y1
