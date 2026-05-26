from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

STATE_OTHER = "other"
STATE_SONG_SELECT = "song_select"
STATE_PLAYING = "playing"
STATE_RESULTS = "results"

_PIPELINE_NODES = {
    STATE_SONG_SELECT: "RhythmSceneOnSongSelect",
    STATE_PLAYING: "RhythmSceneOnPlaying",
    STATE_RESULTS: "RhythmSceneOnResults",
}


class SceneGate:

    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg.get("scene") or {}
        self._state_confirm_frames = max(1, int(sc.get("state_confirm_frames", 2)))
        self._playing_check_interval = max(1, int(sc.get("playing_check_interval", 5)))

        self._state: str = STATE_OTHER
        self._target_state: str = STATE_OTHER
        self._state_streak: int = 0
        self._frame_count: int = 0

    @property
    def state(self) -> str:
        return self._state

    def step(
        self,
        context,
        frame_bgr,
    ) -> tuple[str, dict[str, Any]]:
        self._frame_count += 1

        if self._state == STATE_PLAYING:
            if self._frame_count % self._playing_check_interval != 0:
                return self._state, {
                    "state": self._state,
                    "armed": True,
                    "state_transitioned": False,
                }
            rs_ok = self._check_node(context, STATE_RESULTS, frame_bgr)
            if rs_ok:
                target = STATE_RESULTS
            else:
                return self._state, {
                    "state": self._state,
                    "armed": True,
                    "state_transitioned": False,
                }
        else:
            ss_ok = self._check_node(context, STATE_SONG_SELECT, frame_bgr)
            rs_ok = self._check_node(context, STATE_RESULTS, frame_bgr)
            pl_ok = self._check_node(context, STATE_PLAYING, frame_bgr)

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

    @staticmethod
    def _check_node(context, state: str, frame_bgr) -> bool:
        node = _PIPELINE_NODES.get(state)
        if node is None:
            return False
        result = context.run_recognition(node, frame_bgr)
        return result is not None and result.hit
