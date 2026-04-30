from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_scene_templates, list_song_templates

logger = logging.getLogger(__name__)

_SEL_IDLE = "idle"
_SEL_CLICKING_START = "clicking_start"
_SEL_SEARCHING = "searching"
_SEL_SCROLLING = "scrolling"
_SEL_CLICKING_SONG = "clicking_song"
_SEL_DONE = "done"
_SEL_FAILED = "failed"

_start_template_cache: NDArray[np.uint8] | None = None
_start_template_loaded: bool = False

_song_template_cache: dict[str, NDArray[np.uint8]] = {}


def _load_start_template_once() -> NDArray[np.uint8] | None:
    global _start_template_cache, _start_template_loaded
    if _start_template_loaded:
        return _start_template_cache
    _start_template_loaded = True

    templates = list_scene_templates("song_select")
    for stem, path in templates:
        if stem == "start":
            img = _read_image(path)
            if img is not None:
                _start_template_cache = img
                th, tw = img.shape[:2]
                logger.info("已加载「开始演奏」按钮模板：%s (%dx%d)", path.name, tw, th)
                return img
    logger.warning("未找到「开始演奏」按钮模板：start.png（请放入 scene_templates/song_select/）")
    return None


def _load_song_template_once(name: str) -> NDArray[np.uint8] | None:
    if name in _song_template_cache:
        return _song_template_cache[name]

    templates = list_song_templates()
    for stem, path in templates:
        if stem == name:
            img = _read_image(path)
            if img is not None:
                _song_template_cache[name] = img
                th, tw = img.shape[:2]
                logger.info("已加载歌曲模板：%s (%dx%d)", path.name, tw, th)
                return img
            else:
                logger.warning("无法读取歌曲模板图片：%s", path)
                return None

    available = [s for s, _ in templates]
    logger.warning("未找到歌曲模板: %s (可用: %s)", name, available)
    return None


def _read_image(p: Path) -> NDArray[np.uint8] | None:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        img_bytes = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img


_DEFAULT_SONG = "Heroic_Appearance"


class SongSelector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg.get("song_select") or {}
        self._song_select_enabled = bool(sc.get("enabled", False))
        self._song_name = str(sc.get("song_name", ""))
        self._scroll_area_x_frac = float(sc.get("scroll_area_x_frac", 0.25))
        self._scroll_area_y_frac = float(sc.get("scroll_area_y_frac", 0.50))
        self._scroll_delta = int(sc.get("scroll_delta", -3))
        self._max_scroll_attempts = max(1, int(sc.get("max_scroll_attempts", 30)))
        self._match_threshold = float(sc.get("match_threshold", 0.75))
        self._click_delay = float(sc.get("click_delay_sec", 0.5))
        self._start_delay = float(sc.get("start_delay_sec", 0.8))
        self._start_match_threshold = float(sc.get("start_match_threshold", 0.75))
        self._max_start_retries = max(1, int(sc.get("max_start_retries", 5)))

        self._template: NDArray[np.uint8] | None = None
        self._start_template: NDArray[np.uint8] | None = None
        self._state: str = _SEL_IDLE
        self._scroll_attempts: int = 0
        self._last_action_time: float = 0.0
        self._match_loc: tuple[int, int] | None = None
        self._start_retry_count: int = 0

        self._start_template = _load_start_template_once()

        if not self._song_name:
            self._song_name = _DEFAULT_SONG
            logger.info("未指定歌曲，默认选择: %s", self._song_name)

        self._template = _load_song_template_once(self._song_name)
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
        tpl = _load_song_template_once(name)
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
            match = self._find_template(frame_bgr, self._template, self._match_threshold)
            if match is not None:
                self._match_loc = match
                self._state = _SEL_CLICKING_SONG
                logger.info(
                    "歌曲模板匹配成功: 位置=(%d,%d), 将点击选中",
                    match[0], match[1],
                )
            elif self._scroll_attempts < self._max_scroll_attempts:
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
            if scroll_func is not None:
                sx = int(self._scroll_area_x_frac * w)
                sy = int(self._scroll_area_y_frac * h)
                scroll_func(sx, sy, self._scroll_delta)
            self._scroll_attempts += 1
            self._last_action_time = now
            self._state = _SEL_SEARCHING
            logger.debug("滚动搜索: 第 %d 次", self._scroll_attempts)
            return {"state": self._state, "action": "scroll", "scroll_attempts": self._scroll_attempts}

        if self._state == _SEL_CLICKING_SONG:
            if now - self._last_action_time < self._click_delay:
                return {"state": self._state, "action": "waiting"}
            if self._match_loc is not None:
                mx, my = self._match_loc
                controller.post_click(mx, my).wait()
                self._last_action_time = now
                self._start_retry_count = 0
                self._state = _SEL_CLICKING_START
                logger.info("已点击目标歌曲位置 (%d,%d)，等待后点击开始演奏", mx, my)
            else:
                self._state = _SEL_SEARCHING
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
                logger.info("已点击「开始演奏」按钮 (模板匹配位置 %d,%d)", sx, sy)
            else:
                self._start_retry_count += 1
                if self._start_retry_count < self._max_start_retries:
                    self._last_action_time = now
                    logger.debug(
                        "未匹配到「开始演奏」按钮，重试 %d/%d",
                        self._start_retry_count, self._max_start_retries,
                    )
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
    ) -> tuple[int, int] | None:
        th, tw = tpl.shape[:2]
        fh, fw = frame_bgr.shape[:2]
        if th > fh or tw > fw:
            return None
        result = cv2.matchTemplate(frame_bgr, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            return cx, cy
        return None
