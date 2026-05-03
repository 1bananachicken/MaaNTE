import json
import logging
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from ..utils.config import load_rhythm_config
from ..utils.lanes import build_lane_layout, LaneLayout
from ..utils.detector import DrumDetector
from ..utils.assets import list_scene_templates, read_image


logger = logging.getLogger(__name__)

_LANES = ("d", "f", "j", "k")
_VK = {"d": 0x44, "f": 0x46, "j": 0x4A, "k": 0x4B}

_TIMEOUT_SEC = 600
_SCENE_LOCK_SEC = 8.0
_PLAYING_CHECK_THRESHOLD = 0.7
_MIN_CONFIRM_NOT_PLAYING = 3


def _press_keys(controller, lane_indices: list[int], key_hold_sec: float = 0.01):
    for li in lane_indices:
        controller.post_key_down(_VK[_LANES[li]])
    time.sleep(key_hold_sec)
    for li in lane_indices:
        controller.post_key_up(_VK[_LANES[li]])


def _load_playing_check_template() -> NDArray[np.uint8] | None:
    templates = list_scene_templates("playing")
    for name, path in templates:
        if name == "pause":
            img = read_image(path)
            if img is not None:
                logger.info(
                    "已加载演奏状态检测模板: pause.png (%dx%d)",
                    img.shape[1],
                    img.shape[0],
                )
                return img
    logger.warning("未找到演奏状态检测模板 (pause.png)")
    return None


def _is_still_playing(
    frame: NDArray[np.uint8], template: NDArray[np.uint8] | None
) -> bool:
    if template is None or frame is None or frame.size == 0:
        return True
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    if th > fh or tw > fw:
        return True
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= _PLAYING_CHECK_THRESHOLD


@AgentServer.custom_action("auto_rhythm_play")
class AutoRhythmPlay(CustomAction):
    _playing_check_template: NDArray[np.uint8] | None = None
    _template_loaded: bool = False

    @classmethod
    def _ensure_template_loaded(cls) -> None:
        if cls._template_loaded:
            return
        cls._template_loaded = True
        cls._playing_check_template = _load_playing_check_template()

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self._ensure_template_loaded()
        controller = context.tasker.controller
        cfg = load_rhythm_config()

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                params = {}
            if "target_fps" in params:
                cfg.setdefault("run", {})["target_fps"] = int(params["target_fps"])

        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)
        key_hold_sec = float(cfg.get("keys", {}).get("key_hold_sec", 0.01))

        scene_lock_sec = float(
            cfg.get("scene", {}).get("scene_lock_timeout_sec", _SCENE_LOCK_SEC)
        )
        confirm_not_playing = max(
            _MIN_CONFIRM_NOT_PLAYING,
            int(
                cfg.get("scene", {}).get("state_confirm_frames", _MIN_CONFIRM_NOT_PLAYING)
            ),
        )

        detector = DrumDetector(cfg)
        drum_available = detector.available
        if not drum_available:
            logger.warning("鼓面模板缺失，演奏检测不可用")

        logger.info(
            "演奏开始 | FPS=%d | 鼓面检测=%s | 场景冷却=%.1fs(音符命中重置) | 退出确认=%d帧",
            target_fps, drum_available, scene_lock_sec, confirm_not_playing,
        )

        start_time = time.perf_counter()
        frame_count = 0
        not_playing_streak = 0
        cached_layout: LaneLayout | None = None
        cached_layout_size: tuple[int, int] = (0, 0)
        playing_tpl = self._playing_check_template

        scene_lock_until = time.perf_counter() + scene_lock_sec

        while True:
            if context.tasker.stopping:
                logger.info("tasker 停止信号，退出演奏")
                return CustomAction.RunResult(success=False)

            elapsed_total = time.perf_counter() - start_time
            if elapsed_total > _TIMEOUT_SEC:
                logger.warning("演奏超时 (%d秒)", _TIMEOUT_SEC)
                return CustomAction.RunResult(success=False)

            t0 = time.perf_counter()

            controller.post_screencap().wait()
            frame = controller.cached_image
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue

            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            fh, fw = frame.shape[:2]
            if fh <= 0 or fw <= 0:
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.perf_counter()

            if drum_available:
                if (fw, fh) != cached_layout_size:
                    cached_layout = build_lane_layout(cfg, fw, fh)
                    cached_layout_size = (fw, fh)
                triggers, scores = detector.analyze(frame, cached_layout)
                triggered_lanes = [i for i, t in enumerate(triggers) if t]
                if triggered_lanes:
                    scene_lock_until = now + scene_lock_sec
                    not_playing_streak = 0
                    _press_keys(controller, triggered_lanes, key_hold_sec)

            if now >= scene_lock_until:
                if not _is_still_playing(frame, playing_tpl):
                    not_playing_streak += 1
                    if not_playing_streak >= confirm_not_playing:
                        logger.info(
                            "演奏结束 (帧#%d, 耗时%.1f秒, 连续未匹配%d帧)",
                            frame_count, elapsed_total, not_playing_streak,
                        )
                        return CustomAction.RunResult(success=True)
                else:
                    not_playing_streak = 0

            elapsed_frame = time.perf_counter() - t0
            if elapsed_frame < frame_interval:
                time.sleep(frame_interval - elapsed_frame)