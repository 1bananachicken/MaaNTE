from __future__ import annotations

import logging
import time
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_scene_templates, list_song_templates, read_image

logger = logging.getLogger(__name__)

_SEL_IDLE = "idle"
_SEL_CLICKING_START = "clicking_start"
_SEL_SEARCHING = "searching"
_SEL_SCROLLING = "scrolling"
_SEL_CLICKING_SONG = "clicking_song"
_SEL_DONE = "done"
_SEL_FAILED = "failed"

_DEFAULT_SONG = "Heroic_Appearance"
_AUTO_SELECT_SONG = "迷星叫"


class SongSelector:
    _start_template_cache: NDArray[np.uint8] | None = None
    _start_template_loaded: bool = False
    _song_template_cache: dict[str, NDArray[np.uint8]] = {}

    @classmethod
    def _load_start_template(cls) -> NDArray[np.uint8] | None:
        if cls._start_template_loaded:
            return cls._start_template_cache
        cls._start_template_loaded = True

        templates = list_scene_templates("song_select")
        for stem, path in templates:
            if stem == "start":
                img = read_image(path)
                if img is not None:
                    cls._start_template_cache = img
                    return img
        logger.warning("未找到「开始演奏」按钮模板：start.png（请放入 scene_templates/song_select/）")
        return None

    @classmethod
    def _load_song_template(cls, name: str) -> NDArray[np.uint8] | None:
        if name in cls._song_template_cache:
            return cls._song_template_cache[name]

        templates = list_song_templates()
        for stem, path in templates:
            if stem == name:
                img = read_image(path)
                if img is not None:
                    cls._song_template_cache[name] = img
                    th, tw = img.shape[:2]
                    return img
                else:
                    logger.warning("无法读取歌曲模板图片：%s", path)
                    return None

        available = [s for s, _ in templates]
        logger.warning("未找到歌曲模板: %s (可用: %s)", name, available)
        return None

    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg.get("song_select") or {}
        self._song_select_enabled = bool(sc.get("enabled", False))
        self._auto_select = bool(sc.get("auto_select", False))
        self._song_name = str(sc.get("song_name", ""))
        song_list_roi_list = sc.get("song_list_roi", [47, 117, 550, 510])
        if isinstance(song_list_roi_list, list) and len(song_list_roi_list) == 4:
            self._song_list_roi = tuple(int(v) for v in song_list_roi_list)
        else:
            self._song_list_roi = (47, 117, 550, 510)
        self._scroll_area_x_frac = float(sc.get("scroll_area_x_frac", 0.25))
        self._scroll_area_y_frac = float(sc.get("scroll_area_y_frac", 0.50))
        self._scroll_delta = int(sc.get("scroll_delta", -3))
        self._max_scroll_attempts = max(1, int(sc.get("max_scroll_attempts", 30)))
        self._match_threshold = float(sc.get("match_threshold", 0.75))
        self._click_delay = float(sc.get("click_delay_sec", 0.5))
        self._start_delay = float(sc.get("start_delay_sec", 0.8))
        self._start_match_threshold = float(sc.get("start_match_threshold", 0.75))
        self._max_start_retries = max(1, int(sc.get("max_start_retries", 5)))
        self._scroll_settle_delay = float(sc.get("scroll_settle_delay_sec", 0.4))
        self._click_reverify_threshold = float(sc.get("click_reverify_threshold", 0.70))
        self._max_click_reverify_retries = max(1, int(sc.get("max_click_reverify_retries", 2)))

        self._template: NDArray[np.uint8] | None = None
        self._start_template: NDArray[np.uint8] | None = None
        self._state: str = _SEL_IDLE
        self._scroll_attempts: int = 0
        self._last_action_time: float = 0.0
        self._match_loc: tuple[int, int] | None = None
        self._start_retry_count: int = 0
        self._consecutive_down_fails: int = 0
        self._one_time_ds: int | None = None
        self._post_scroll_time: float = 0.0
        self._click_reverify_retries: int = 0

        self._start_template = self._load_start_template()

        if self._auto_select:
            self._song_name = _AUTO_SELECT_SONG
            logger.info("自动选曲启用，选择: %s", self._song_name)
        elif not self._song_name:
            self._song_name = _DEFAULT_SONG
            logger.info("未指定歌曲，默认选择: %s", self._song_name)

        self._template = self._load_song_template(self._song_name)
        if self._template is not None:
            self._song_select_enabled = True

    @property
    def enabled(self) -> bool:
        return self._song_select_enabled

    @property
    def song_name(self) -> str:
        return self._song_name

    def select_song(self, name: str) -> bool:
        self._song_name = name
        tpl = self._load_song_template(name)
        if tpl is not None:
            self._template = tpl
            self._song_select_enabled = True
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._state = _SEL_IDLE
        self._scroll_attempts = 0
        self._last_action_time = 0.0
        self._match_loc = None
        self._start_retry_count = 0
        self._consecutive_down_fails = 0
        self._one_time_ds = None
        self._post_scroll_time = 0.0
        self._click_reverify_retries = 0

    @property
    def state(self) -> str:
        return self._state

    def step(
        self,
        frame_bgr: NDArray[np.uint8],
        controller: Any,
        scroll_func: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        if self._state == _SEL_IDLE:
            self._start_retry_count = 0
            self._last_action_time = time.perf_counter()
            self._state = _SEL_SEARCHING
            self._scroll_attempts = 0

        if self._state == _SEL_SEARCHING:
            if self._scroll_attempts > 0 and now - self._post_scroll_time < self._scroll_settle_delay:
                return {"state": self._state, "action": "settling", "scroll_attempts": self._scroll_attempts}
            match = self._find_template(frame_bgr, self._template, self._match_threshold, self._song_list_roi)
            if match is not None:
                self._match_loc = match
                self._consecutive_down_fails = 0
                self._one_time_ds = None
                self._click_reverify_retries = 0
                self._state = _SEL_CLICKING_SONG
            elif self._scroll_attempts < self._max_scroll_attempts:
                if self._consecutive_down_fails >= 5 and self._one_time_ds is None:
                    self._one_time_ds = 1
                    self._consecutive_down_fails = 0
                self._state = _SEL_SCROLLING
            else:
                self._state = _SEL_FAILED
                logger.warning(
                    "已滚动 %d 次仍未找到目标歌曲，选歌失败",
                    self._scroll_attempts,
                )
            return {"state": self._state, "scroll_attempts": self._scroll_attempts}

        if self._state == _SEL_SCROLLING:
            if now - self._last_action_time < self._click_delay:
                return {"state": self._state, "action": "waiting"}
            direction = self._one_time_ds if self._one_time_ds is not None else self._scroll_delta
            if self._one_time_ds is not None:
                self._one_time_ds = None
            else:
                self._consecutive_down_fails += 1
            if scroll_func is not None:
                sx = self._song_list_roi[0] + self._song_list_roi[2] // 2
                sy = self._song_list_roi[1] + self._song_list_roi[3] // 2
                scroll_func(sx, sy, direction)
            self._scroll_attempts += 1
            self._last_action_time = now
            self._post_scroll_time = time.perf_counter()
            self._state = _SEL_SEARCHING
            return {"state": self._state, "action": "scroll", "scroll_attempts": self._scroll_attempts}

        if self._state == _SEL_CLICKING_SONG:
            if now - self._last_action_time < self._click_delay:
                return {"state": self._state, "action": "waiting"}
            current_match = self._find_template(frame_bgr, self._template, self._click_reverify_threshold, self._song_list_roi)
            if current_match is None:
                self._click_reverify_retries += 1
                if self._click_reverify_retries < self._max_click_reverify_retries:
                    logger.warning(
                        "点击前重验证失败 (%d/%d)，歌单可能已回滚，重新搜索",
                        self._click_reverify_retries, self._max_click_reverify_retries,
                    )
                    self._last_action_time = now
                    self._post_scroll_time = 0.0
                    self._state = _SEL_SEARCHING
                    return {"state": self._state, "action": "reverify_fail"}
                else:
                    logger.warning(
                        "重验证 %d 次均失败，回退到滚动搜索",
                        self._click_reverify_retries,
                    )
                    self._click_reverify_retries = 0
                    self._last_action_time = now
                    self._post_scroll_time = 0.0
                    self._state = _SEL_SEARCHING
                    return {"state": self._state, "action": "reverify_fail"}
            self._match_loc = current_match
            mx, my = current_match
            controller.post_click(mx, my).wait()
            self._last_action_time = now
            self._start_retry_count = 0
            self._state = _SEL_CLICKING_START
            return {"state": self._state, "action": "click_song"}

        if self._state == _SEL_CLICKING_START:
            if now - self._last_action_time < self._start_delay:
                return {"state": self._state, "action": "waiting"}
            if self._start_template is not None:
                start_loc = self._find_template(frame_bgr, self._start_template, self._start_match_threshold)
            else:
                start_loc = None
            if start_loc is not None:
                sx, sy = start_loc
                controller.post_click(sx, sy).wait()
                self._last_action_time = now
                self._state = _SEL_DONE
            else:
                self._start_retry_count += 1
                if self._start_retry_count < self._max_start_retries:
                    self._last_action_time = now
                else:
                    logger.warning("未匹配到「开始演奏」按钮 (已重试 %d 次)，选歌失败", self._start_retry_count)
                    self._state = _SEL_FAILED
            return {"state": self._state, "action": "click_start"}

        if self._state == _SEL_DONE:
            return {"state": self._state, "action": "done"}

        if self._state == _SEL_FAILED:
            return {"state": self._state, "action": "failed"}

        return {"state": self._state, "action": "unknown"}

    def _find_template(
        self,
        frame_bgr: NDArray[np.uint8],
        tpl: NDArray[np.uint8],
        threshold: float,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int] | None:
        th, tw = tpl.shape[:2]
        fh, fw = frame_bgr.shape[:2]
        if th > fh or tw > fw:
            return None
        search_area = frame_bgr
        offset_x, offset_y = 0, 0
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = max(0, min(rx, fw))
            ry = max(0, min(ry, fh))
            rw = min(rw, fw - rx)
            rh = min(rh, fh - ry)
            if rw < tw or rh < th:
                return None
            search_area = frame_bgr[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry
        result = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            cx = max_loc[0] + tw // 2 + offset_x
            cy = max_loc[1] + th // 2 + offset_y
            return cx, cy
        return None
