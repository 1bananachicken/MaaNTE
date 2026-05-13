from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_scene_templates, read_image

logger = logging.getLogger(__name__)

STATE_OTHER = "other"
STATE_SONG_SELECT = "song_select"
STATE_PLAYING = "playing"
STATE_RESULTS = "results"


class SceneGate:
    _template_cache: dict[str, list[tuple[str, NDArray[np.uint8]]]] = {}

    @classmethod
    def _load_scene_templates(cls, kind: str) -> list[tuple[str, NDArray[np.uint8]]]:
        if kind in cls._template_cache:
            return cls._template_cache[kind]

        templates: list[tuple[str, NDArray[np.uint8]]] = []
        for name, tpl_path in list_scene_templates(kind):
            img = read_image(tpl_path)
            if img is None:
                logger.warning("无法读取场景模板：%s", tpl_path)
                continue
            templates.append((name, img))
        if not templates:
            logger.debug("未找到场景模板：%s", kind)

        cls._template_cache[kind] = templates
        return templates

    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg.get("scene") or {}
        self._song_select_thresh = float(sc.get("song_select_match_threshold", 0.75))
        self._results_thresh = float(sc.get("results_match_threshold", 0.75))
        self._playing_thresh = float(sc.get("playing_match_threshold", 0.75))
        self._state_confirm_frames = max(1, int(sc.get("state_confirm_frames", 2)))
        self._match_vote_min = max(1, int(sc.get("match_vote_min", 1)))
        self._playing_check_interval = max(1, int(sc.get("playing_check_interval", 5)))

        template_roi_cfg = sc.get("template_roi") or {}
        self._template_roi: dict[str, dict[str, tuple[int, int, int, int]]] = {}
        for kind, name_map in template_roi_cfg.items():
            self._template_roi[kind] = {}
            for name, roi_list in name_map.items():
                if isinstance(roi_list, list) and len(roi_list) == 4:
                    self._template_roi[kind][name] = tuple(int(v) for v in roi_list)

        self._song_select_tpls = self._load_scene_templates("song_select")
        self._results_tpls = self._load_scene_templates("results")
        self._playing_tpls = self._load_scene_templates("playing")

        self._has_any_templates = bool(
            self._song_select_tpls or self._results_tpls or self._playing_tpls
        )

        self._state: str = STATE_OTHER
        self._target_state: str = STATE_OTHER
        self._state_streak: int = 0
        self._frame_count: int = 0

    @property
    def state(self) -> str:
        return self._state

    def step(
        self,
        frame_bgr: NDArray[np.uint8],
    ) -> tuple[str, dict[str, Any]]:
        self._frame_count += 1

        if not self._has_any_templates:
            return STATE_PLAYING, {
                "state": STATE_PLAYING,
                "armed": True,
                "state_transitioned": False,
            }

        if self._state == STATE_PLAYING:
            if self._frame_count % self._playing_check_interval != 0:
                return self._state, {
                    "state": self._state,
                    "armed": True,
                    "state_transitioned": False,
                }
            rs_ok, rs_val = self._vote(frame_bgr, self._results_tpls, self._results_thresh, "results")
            if rs_ok:
                target = STATE_RESULTS
            else:
                return self._state, {
                    "state": self._state,
                    "armed": True,
                    "state_transitioned": False,
                }
        else:
            ss_ok, ss_val = self._vote(frame_bgr, self._song_select_tpls, self._song_select_thresh, "song_select")
            rs_ok, rs_val = self._vote(frame_bgr, self._results_tpls, self._results_thresh, "results")
            pl_ok, pl_val = self._vote(frame_bgr, self._playing_tpls, self._playing_thresh, "playing")

            if ss_ok:
                target = STATE_SONG_SELECT
            elif rs_ok:
                target = STATE_RESULTS
            elif pl_ok:
                target = STATE_PLAYING
            else:
                target = STATE_OTHER

        if target != self._target_state:
            self._target_state = target
            self._state_streak = 1
        else:
            self._state_streak += 1

        prev_state = self._state
        if target != self._state and self._state_streak >= self._state_confirm_frames:
            self._state = target
            self._state_streak = 0

        state_transitioned = prev_state != self._state
        armed = self._state == STATE_PLAYING

        info: dict[str, Any] = {
            "state": self._state,
            "armed": armed,
            "state_transitioned": state_transitioned,
        }
        return self._state, info

    def _vote(
        self,
        frame_bgr: NDArray[np.uint8],
        templates: list[tuple[str, NDArray[np.uint8]]],
        threshold: float,
        kind: str = "",
    ) -> tuple[bool, float]:
        if not templates:
            return False, 0.0
        best_val = 0.0
        vote_count = 0
        required_votes = min(self._match_vote_min, len(templates))
        fh, fw = frame_bgr.shape[:2]
        roi_map = self._template_roi.get(kind, {})
        for name, tpl in templates:
            th, tw = tpl.shape[:2]
            if th <= 0 or tw <= 0 or th > fh or tw > fw:
                continue
            roi = roi_map.get(name)
            search_area = frame_bgr
            if roi is not None:
                rx, ry, rw, rh = roi
                rx = max(0, min(rx, fw))
                ry = max(0, min(ry, fh))
                rw = min(rw, fw - rx)
                rh = min(rh, fh - ry)
                if rw < tw or rh < th:
                    continue
                search_area = frame_bgr[ry:ry + rh, rx:rx + rw]
            result = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = float(max_val)
            if max_val >= threshold:
                vote_count += 1
                if vote_count >= required_votes:
                    return True, best_val
        return False, best_val
