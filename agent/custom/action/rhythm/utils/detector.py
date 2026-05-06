from __future__ import annotations

from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_drum_templates, _get_image_root, read_image
from .lanes import LaneLayout

logger = logging.getLogger(__name__)

_LANE_NAMES = ("d", "f", "j", "k")


@dataclass(frozen=True)
class DrumCandidate:
    score: float
    center_x: float
    center_y: float


class DrumDetector:
    _cache: dict[int, NDArray[np.uint8]] | None = None

    @classmethod
    def _load_drum_templates(cls) -> dict[int, NDArray[np.uint8]]:
        if cls._cache is not None:
            return cls._cache

        cache: dict[int, NDArray[np.uint8]] = {}
        available = list_drum_templates()
        name_to_idx = {name: i for i, name in enumerate(_LANE_NAMES)}
        for stem, path in available:
            key = stem.lower().replace("press_", "")
            idx = name_to_idx.get(key)
            if idx is None:
                continue
            img = read_image(path)
            if img is not None:
                cache[idx] = img
                th, tw = img.shape[:2]
                logger.info("已加载鼓面模板: %s (%dx%d)", path.name, tw, th)

        loaded_count = len(cache)
        if loaded_count == 0:
            logger.warning(
                "未找到任何鼓面模板图片，请将 press_d.png / press_f.png / press_j.png / press_k.png "
                "放入 %s",
                _get_image_root() / "drum_templates",
            )
        elif loaded_count < 4:
            missing = [_LANE_NAMES[i] for i in range(4) if i not in cache]
            logger.warning("部分鼓面模板缺失: %s，对应轨道将无法检测", missing)

        cls._cache = cache
        return cache

    def __init__(self, cfg: dict[str, Any]) -> None:
        tcfg = cfg.get("template_detection") or {}
        self._thresholds: list[float] = tcfg.get("thresholds", [0.81, 0.80, 0.80, 0.81])
        if len(self._thresholds) < 4:
            self._thresholds.extend([0.80] * (4 - len(self._thresholds)))
        self._region_extend_up_frac = float(tcfg.get("region_extend_up_frac", 0.14))
        self._region_extend_down_frac = float(tcfg.get("region_extend_down_frac", 0.15))
        self._region_width_multiplier = float(tcfg.get("region_width_multiplier", 4.0))
        self._candidate_threshold_margin = max(
            0.0,
            float(tcfg.get("candidate_threshold_margin", 0.03)),
        )
        self._candidate_nms_distance_px = max(
            1.0,
            float(tcfg.get("candidate_nms_distance_px", 30.0)),
        )
        self._max_candidates_per_lane = max(
            1,
            int(tcfg.get("max_candidates_per_lane", 4)),
        )
        enabled = tcfg.get("enabled_lanes")
        if isinstance(enabled, list) and len(enabled) == 4:
            self._enabled_lanes = [bool(x) for x in enabled]
        else:
            self._enabled_lanes = [True, True, True, True]

        self._templates: list[NDArray[np.uint8] | None] = [None, None, None, None]
        self._template_loaded: list[bool] = [False, False, False, False]

        cache = self._load_drum_templates()
        for idx, img in cache.items():
            self._templates[idx] = img
            self._template_loaded[idx] = True

        self._executor = ThreadPoolExecutor(max_workers=4)

    @property
    def available(self) -> bool:
        return any(self._template_loaded)

    def _match_lane(
        self,
        lane_idx: int,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[float, list[DrumCandidate]]:
        if not self._enabled_lanes[lane_idx] or not self._template_loaded[lane_idx] or self._templates[lane_idx] is None:
            return 0.0, []

        tpl = self._templates[lane_idx]
        th, tw = tpl.shape[:2]

        cx = layout.center_x[lane_idx]
        base_half = max(layout.half_width_px, tw // 2)
        half_w = int(round(base_half * self._region_width_multiplier))
        half_w = max(half_w, tw // 2 + 2)
        jy0 = layout.judge_y0_by_lane[lane_idx]
        jy1 = layout.judge_y1_by_lane[lane_idx]
        extend_up = int(self._region_extend_up_frac * layout.frame_h)
        extend_down = max(4, int(self._region_extend_down_frac * layout.frame_h))

        rx0 = max(0, cx - half_w)
        rx1 = min(layout.frame_w, cx + half_w)
        ry0 = max(0, jy0 - extend_up)
        ry1 = min(layout.frame_h, jy1 + extend_down)

        if rx1 - rx0 < tw or ry1 - ry0 < th:
            return 0.0, []

        roi = frame_bgr[ry0:ry1, rx0:rx1]
        result = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        candidate_threshold = self._thresholds[lane_idx] - self._candidate_threshold_margin
        ys, xs = np.where(result >= candidate_threshold)

        candidates: list[DrumCandidate] = []
        if len(xs) == 0:
            center_x = float(rx0 + max_loc[0] + tw / 2.0)
            center_y = float(ry0 + max_loc[1] + th / 2.0)
            return float(max_val), [DrumCandidate(float(max_val), center_x, center_y)]

        raw_candidates = [
            DrumCandidate(
                score=float(result[y, x]),
                center_x=float(rx0 + x + tw / 2.0),
                center_y=float(ry0 + y + th / 2.0),
            )
            for y, x in zip(ys, xs)
        ]
        raw_candidates.sort(key=lambda c: c.score, reverse=True)

        for candidate in raw_candidates:
            if len(candidates) >= self._max_candidates_per_lane:
                break
            duplicate = any(
                abs(candidate.center_y - kept.center_y) < self._candidate_nms_distance_px
                for kept in candidates
            )
            if not duplicate:
                candidates.append(candidate)

        # logger.debug(
        #     "轨道 %s: max_val=%.3f, threshold=%.3f, triggered=%s, roi_size=(%dx%d)",
        #     _LANE_NAMES[lane_idx], max_val, self._thresholds[lane_idx],
        #     max_val >= self._thresholds[lane_idx], roi.shape[1], roi.shape[0]
        # )

        return float(max_val), candidates

    def analyze(
        self,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[list[float], list[list[DrumCandidate]]]:
        scores: list[float] = [0.0, 0.0, 0.0, 0.0]
        candidates_by_lane: list[list[DrumCandidate]] = [[], [], [], []]

        futures = {}
        for i in range(4):
            future = self._executor.submit(self._match_lane, i, frame_bgr, layout)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            score, candidates = future.result()
            scores[idx] = score
            candidates_by_lane[idx] = candidates

        return scores, candidates_by_lane
