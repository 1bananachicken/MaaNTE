from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_drum_templates, _get_image_root, read_image
from .lanes import LaneLayout

logger = logging.getLogger(__name__)

_LANE_NAMES = ("d", "f", "j", "k")


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
        enabled = tcfg.get("enabled_lanes")
        if isinstance(enabled, list) and len(enabled) == 4:
            self._enabled_lanes = [bool(x) for x in enabled]
        else:
            self._enabled_lanes = [True, True, True, True]

        cooldown_global = float(tcfg.get("cooldown_sec", 0.05))
        cooldown_by_lane = tcfg.get("cooldown_sec_by_lane")
        if isinstance(cooldown_by_lane, list) and len(cooldown_by_lane) == 4:
            self._cooldown_sec: list[float] = [float(x) for x in cooldown_by_lane]
        else:
            self._cooldown_sec = [cooldown_global] * 4

        self._last_trigger_time: list[float] = [0.0] * 4

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
    ) -> tuple[bool, float]:
        if not self._enabled_lanes[lane_idx] or not self._template_loaded[lane_idx] or self._templates[lane_idx] is None:
            return False, 0.0

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
            return False, 0.0

        roi = frame_bgr[ry0:ry1, rx0:rx1]
        result = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val >= self._thresholds[lane_idx]:
            return True, float(max_val)

        return False, float(max_val)

    def analyze(
        self,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[list[bool], list[float]]:
        triggers: list[bool] = [False, False, False, False]
        scores: list[float] = [0.0, 0.0, 0.0, 0.0]

        now = time.perf_counter()
        lanes_to_check: list[int] = []
        for i in range(4):
            if now - self._last_trigger_time[i] < self._cooldown_sec[i]:
                continue
            lanes_to_check.append(i)

        if not lanes_to_check:
            return triggers, scores

        futures = {}
        for i in lanes_to_check:
            future = self._executor.submit(self._match_lane, i, frame_bgr, layout)
            futures[future] = i

        for future in futures:
            idx = futures[future]
            triggered, score = future.result()
            if triggered:
                triggers[idx] = True
                scores[idx] = score
                self._last_trigger_time[idx] = now
            else:
                scores[idx] = score

        return triggers, scores
