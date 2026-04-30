from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LaneLayout:
    frame_w: int
    frame_h: int
    center_x: list[int]
    half_width_px: int
    judge_y0_by_lane: list[int]
    judge_y1_by_lane: list[int]


def _frac(cfg: dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    return float(v)


def _center_x_at_judge(
    top_x: int, bottom_x: int, judge_y: int, frame_h: int,
) -> int:
    t = judge_y / float(frame_h) if frame_h > 0 else 0.5
    t = max(0.0, min(1.0, t))
    return int(round(top_x + (bottom_x - top_x) * t))


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

    half_band = max(2, int(round(band_half * frame_h)))
    half_width_px = max(2, int(round(half_w_frac * frame_w)))
    bottom_center_x = [int(round(c * frame_w)) for c in centers]
    top_center_x = [int(round(c * frame_w)) for c in top_centers]

    judge_y0_by_lane: list[int] = []
    judge_y1_by_lane: list[int] = []
    center_x: list[int] = []
    for i in range(4):
        lcy = int(round(float(judge_y_by_lane[i]) * frame_h))
        l_full_top = max(0, lcy - half_band)
        l_full_bottom = min(frame_h, lcy + half_band)
        judge_y0_by_lane.append(l_full_top)
        judge_y1_by_lane.append(l_full_bottom)
        center_x.append(_center_x_at_judge(top_center_x[i], bottom_center_x[i], lcy, frame_h))

    return LaneLayout(
        frame_w=frame_w,
        frame_h=frame_h,
        center_x=center_x,
        half_width_px=half_width_px,
        judge_y0_by_lane=judge_y0_by_lane,
        judge_y1_by_lane=judge_y1_by_lane,
    )
