"""基于 HSV 的判定带内音符像素检测。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .lanes import LaneLayout, lane_center_x_at_y, lane_judge_slice

logger = logging.getLogger(__name__)


@dataclass
class LaneDetectState:
    last_fire: float = 0.0
    last_cooldown_log_time: float = 0.0
    recent_component_fires: list[tuple[float, float]] | None = None


def _hsv_mask(
    hsv: NDArray[np.uint8],
    low: tuple[int, int, int],
    high: tuple[int, int, int],
) -> NDArray[np.uint8]:
    return cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))


def _range_to_tuple(r: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    return (
        int(r["h_min"]),
        int(r["h_max"]),
        int(r["s_min"]),
        int(r["s_max"]),
        int(r["v_min"]),
        int(r["v_max"]),
    )


class RhythmDetector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        det = cfg.get("detection") or {}
        self.min_pixels = int(det.get("min_pixels_per_lane", 180))
        raw_by = det.get("min_pixels_by_lane")
        self._min_by_lane: list[int | None]
        if isinstance(raw_by, list) and len(raw_by) > 0:
            self._min_by_lane = []
            for i in range(4):
                if i < len(raw_by) and raw_by[i] is not None:
                    self._min_by_lane.append(int(raw_by[i]))
                else:
                    self._min_by_lane.append(None)
        else:
            self._min_by_lane = [None, None, None, None]
        self.cooldown_sec = float(det.get("cooldown_sec", 0.12))
        raw_cd_by = det.get("cooldown_sec_by_lane")
        if isinstance(raw_cd_by, list) and len(raw_cd_by) > 0:
            self._cooldown_by_lane = [
                float(raw_cd_by[i]) if i < len(raw_cd_by) and raw_cd_by[i] is not None else self.cooldown_sec
                for i in range(4)
            ]
        else:
            self._cooldown_by_lane = [self.cooldown_sec] * 4
        self.morph_k = max(0, int(det.get("morph_kernel", 3)))
        self._log_cooldown_debug = bool(det.get("log_cooldown_debug", False))
        enabled = det.get("enabled_lanes")
        if isinstance(enabled, list) and len(enabled) == 4:
            self._enabled_lanes = [bool(x) for x in enabled]
        else:
            self._enabled_lanes = [True, True, True, True]
        component_lanes = det.get("component_mode_lanes")
        if isinstance(component_lanes, list) and len(component_lanes) == 4:
            self._component_lanes = [bool(x) for x in component_lanes]
        else:
            self._component_lanes = [False, False, False, False]
        self._component_min_pixels = int(det.get("component_min_pixels", 120))
        self._component_min_area_frac = det.get("component_min_area_frac")
        self._component_lookahead_px = int(det.get("component_lookahead_px", 70))
        self._component_lookahead_y_frac = det.get("component_lookahead_y_frac")
        self._component_past_px = int(det.get("component_past_px", 30))
        self._component_past_y_frac = det.get("component_past_y_frac")
        self._component_same_note_px = int(det.get("component_same_note_px", 55))
        self._component_same_note_y_frac = det.get("component_same_note_y_frac")
        self._component_history_sec = float(det.get("component_history_sec", 0.35))

        ranges = cfg.get("hsv_ranges") or []
        if len(ranges) != 4:
            raise ValueError("hsv_ranges 必须为 4 项，对应 D F J K")
        self._ranges = ranges
        self._states = [LaneDetectState() for _ in range(4)]

    def min_pixels_for_lane(self, lane_index: int) -> int:
        o = self._min_by_lane[lane_index] if lane_index < len(self._min_by_lane) else None
        return int(o) if o is not None else self.min_pixels

    def update_fire_times(self, fire_times: dict[int, float]) -> None:
        for lane, t in fire_times.items():
            if 0 <= lane < len(self._states):
                self._states[lane].last_fire = t

    def reserve_fire_times(self, lane_indices: list[int], target_time: float) -> None:
        for lane in lane_indices:
            if 0 <= lane < len(self._states):
                self._states[lane].last_fire = target_time

    def analyze(
        self,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[list[bool], list[NDArray[np.uint8]], list[int]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        now = time.perf_counter()

        disabled_indices = [i for i in range(4) if not self._enabled_lanes[i]]
        simple_indices = [i for i in range(4) if self._enabled_lanes[i] and not self._component_lanes[i]]
        component_indices = [i for i in range(4) if self._enabled_lanes[i] and self._component_lanes[i]]

        results: dict[int, tuple[bool, NDArray[np.uint8], int]] = {}

        for i in disabled_indices:
            results[i] = (False, np.zeros((1, 1), dtype=np.uint8), 0)

        for i in simple_indices:
            results[i] = self._detect_simple(i, hsv, layout, now)

        for i in component_indices:
            results[i] = self._detect_component(i, hsv, layout, now)

        triggers = [results[i][0] for i in range(4)]
        masks = [results[i][1] for i in range(4)]
        pixels = [results[i][2] for i in range(4)]
        return triggers, masks, pixels

    def _detect_simple(
        self,
        i: int,
        hsv: NDArray[np.uint8],
        layout: LaneLayout,
        now: float,
    ) -> tuple[bool, NDArray[np.uint8], int]:
        x0, x1, y0, y1 = lane_judge_slice(layout, i)
        if x1 <= x0 or y1 <= y0:
            return False, np.zeros((1, 1), dtype=np.uint8), 0
        roi_hsv = hsv[y0:y1, x0:x1]
        r = self._ranges[i]
        lo = (r["h_min"], r["s_min"], r["v_min"])
        hi = (r["h_max"], r["s_max"], r["v_max"])
        mask = _hsv_mask(roi_hsv, lo, hi)
        h2_min = r.get("h2_min")
        h2_max = r.get("h2_max")
        if h2_min is not None and h2_max is not None:
            m2 = _hsv_mask(
                roi_hsv,
                (int(h2_min), int(r["s_min"]), int(r["v_min"])),
                (int(h2_max), int(r["s_max"]), int(r["v_max"])),
            )
            mask = cv2.bitwise_or(mask, m2)
        if self.morph_k >= 3:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_k, self.morph_k),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        st = self._states[i]
        thr = self.min_pixels_for_lane(i)
        cooldown_sec = self._cooldown_by_lane[i]
        cnt = int(cv2.countNonZero(mask))
        fire = False
        if cnt >= thr and (now - st.last_fire) >= cooldown_sec:
            fire = True
            st.last_fire = now
        elif (
            self._log_cooldown_debug
            and cnt >= thr
            and (now - st.last_fire) < cooldown_sec
            and (now - st.last_cooldown_log_time) >= 0.15
        ):
            st.last_cooldown_log_time = now
            logger.debug(
                "轨道%d: HSV 匹配像素=%d (>=阈值 %d) 但在冷却中，距上次触发 %.3fs",
                i + 1,
                cnt,
                thr,
                now - st.last_fire,
            )
        return fire, mask, cnt

    def _detect_component(
        self,
        i: int,
        hsv: NDArray[np.uint8],
        layout: LaneLayout,
        now: float,
    ) -> tuple[bool, NDArray[np.uint8], int]:
        x0, x1, judge_y0, judge_y1 = lane_judge_slice(layout, i)
        lookahead_px = self._component_lookahead_px_for(layout)
        past_px = self._component_past_px_for(layout)
        y0 = max(0, judge_y0 - lookahead_px)
        y1 = min(layout.frame_h, judge_y1 + past_px)
        cx = lane_center_x_at_y(layout, i, (y0 + y1) // 2)
        x0 = max(0, cx - layout.half_width_px)
        x1 = min(layout.frame_w, cx + layout.half_width_px)
        if x1 <= x0 or y1 <= y0:
            return False, np.zeros((1, 1), dtype=np.uint8), 0
        roi_hsv = hsv[y0:y1, x0:x1]
        r = self._ranges[i]
        lo = (r["h_min"], r["s_min"], r["v_min"])
        hi = (r["h_max"], r["s_max"], r["v_max"])
        mask = _hsv_mask(roi_hsv, lo, hi)
        h2_min = r.get("h2_min")
        h2_max = r.get("h2_max")
        if h2_min is not None and h2_max is not None:
            m2 = _hsv_mask(
                roi_hsv,
                (int(h2_min), int(r["s_min"]), int(r["v_min"])),
                (int(h2_max), int(r["s_max"]), int(r["v_max"])),
            )
            mask = cv2.bitwise_or(mask, m2)
        if self.morph_k >= 3:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_k, self.morph_k),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        st = self._states[i]
        fire, cnt = self._analyze_components(mask, st, layout, y0, judge_y0, judge_y1, now)
        return fire, mask, cnt

    def _analyze_components(
        self,
        mask: NDArray[np.uint8],
        st: LaneDetectState,
        layout: LaneLayout,
        roi_y0: int,
        judge_y0: int,
        judge_y1: int,
        now: float,
    ) -> tuple[bool, int]:
        if st.recent_component_fires is None:
            st.recent_component_fires = []
        st.recent_component_fires = [
            (cy, t) for cy, t in st.recent_component_fires if (now - t) <= self._component_history_sec
        ]

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        min_pixels = self._component_min_pixels_for(layout)
        past_px = self._component_past_px_for(layout)
        same_note_px = self._component_same_note_px_for(layout)
        best_center_y: float | None = None
        best_area = 0
        total_area = 0

        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            cy_local = float(centroids[label][1])
            comp_top = int(stats[label, cv2.CC_STAT_TOP])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            comp_top_global = roi_y0 + comp_top
            comp_bottom_global = comp_top_global + comp_h
            # 组件底部进入判定带才触发；扩展 ROI 只用于把连续同轨音符拆成独立块。
            if comp_bottom_global < judge_y0 or comp_top_global > judge_y1 + past_px:
                continue
            total_area += area
            cy_global = roi_y0 + cy_local
            already_fired = any(
                abs(cy_global - prev_y) <= same_note_px
                for prev_y, _ in st.recent_component_fires
            )
            if already_fired:
                continue
            if area > best_area:
                best_area = area
                best_center_y = cy_global

        if best_center_y is None:
            return False, total_area

        st.recent_component_fires.append((best_center_y, now))
        st.last_fire = now
        return True, max(total_area, best_area)

    def _component_min_pixels_for(self, layout: LaneLayout) -> int:
        if self._component_min_area_frac is not None:
            return max(8, int(round(float(self._component_min_area_frac) * layout.frame_w * layout.frame_h)))
        return self._component_min_pixels

    def _component_lookahead_px_for(self, layout: LaneLayout) -> int:
        if self._component_lookahead_y_frac is not None:
            return max(0, int(round(float(self._component_lookahead_y_frac) * layout.frame_h)))
        return self._component_lookahead_px

    def _component_past_px_for(self, layout: LaneLayout) -> int:
        if self._component_past_y_frac is not None:
            return max(0, int(round(float(self._component_past_y_frac) * layout.frame_h)))
        return self._component_past_px

    def _component_same_note_px_for(self, layout: LaneLayout) -> int:
        if self._component_same_note_y_frac is not None:
            return max(1, int(round(float(self._component_same_note_y_frac) * layout.frame_h)))
        return self._component_same_note_px
