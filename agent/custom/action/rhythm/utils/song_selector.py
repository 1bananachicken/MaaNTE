from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SEL_IDLE = "idle"
_SEL_CLICKING_START = "clicking_start"
_SEL_SEARCHING = "searching"
_SEL_SCROLLING = "scrolling"
_SEL_CLICKING_SONG = "clicking_song"
_SEL_DONE = "done"
_SEL_FAILED = "failed"

_START_BUTTON_NODE = "RhythmSceneStartButton"
_FIND_SONG_NODE = "RhythmSceneFindSong"


class SongSelector:

    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg.get("song_select") or {}
        self._song_select_enabled = bool(sc.get("enabled", False))
        self._auto_select = bool(sc.get("auto_select", False))
        song_list_roi_list = sc.get("song_list_roi", [47, 117, 550, 510])
        if isinstance(song_list_roi_list, list) and len(song_list_roi_list) == 4:
            self._song_list_roi = [int(v) for v in song_list_roi_list]
        else:
            self._song_list_roi = [47, 117, 550, 510]
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

        self._state: str = _SEL_IDLE
        self._scroll_attempts: int = 0
        self._last_action_time: float = 0.0
        self._match_loc: tuple[int, int] | None = None
        self._start_retry_count: int = 0
        self._consecutive_down_fails: int = 0
        self._one_time_ds: int | None = None
        self._post_scroll_time: float = 0.0
        self._click_reverify_retries: int = 0

        if self._auto_select:
            logger.info("自动选曲启用，将匹配 pipeline 默认歌曲")

        self._song_select_enabled = True

    @property
    def enabled(self) -> bool:
        return self._song_select_enabled

    def select_song(self, name: str) -> bool:
        self._song_select_enabled = True
        self.reset()
        return True

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

    @staticmethod
    def _box_center(box) -> tuple[int, int]:
        return box.x + box.w // 2, box.y + box.h // 2

    def _find_song(self, context, frame, threshold) -> tuple[int, int] | None:
        result = context.run_recognition(_FIND_SONG_NODE, frame)
        if result and result.hit and result.box:
            return self._box_center(result.box)
        return None

    def _find_start(self, context, frame, threshold) -> tuple[int, int] | None:
        result = context.run_recognition(
            _START_BUTTON_NODE, frame,
            pipeline_override={
                _START_BUTTON_NODE: {
                    "recognition": {"param": {"threshold": threshold}}
                }
            },
        )
        if result and result.hit and result.box:
            return self._box_center(result.box)
        return None

    def step(
        self,
        context,
        frame,
        controller: Any,
        scroll_func: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter()

        if self._state == _SEL_IDLE:
            self._start_retry_count = 0
            self._last_action_time = time.perf_counter()
            self._state = _SEL_SEARCHING
            self._scroll_attempts = 0

        if self._state == _SEL_SEARCHING:
            if self._scroll_attempts > 0 and now - self._post_scroll_time < self._scroll_settle_delay:
                return {"state": self._state, "action": "settling", "scroll_attempts": self._scroll_attempts}
            match = self._find_song(context, frame, self._match_threshold)
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
            current_match = self._find_song(context, frame, self._click_reverify_threshold)
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
            start_loc = self._find_start(context, frame, self._start_match_threshold)
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
