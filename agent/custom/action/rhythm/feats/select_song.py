import json
import logging
import time
from typing import Any

import cv2

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from ..utils.config import load_rhythm_config
from ..utils.presence import (
    STATE_SONG_SELECT,
    SceneGate,
)
from ..utils.song_selector import SongSelector


logger = logging.getLogger(__name__)


@AgentServer.custom_action("auto_rhythm_select_song")
class AutoRhythmSelectSong(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        cfg = load_rhythm_config()

        params: dict[str, Any] = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        if "song_name" in params:
            cfg.setdefault("song_select", {})["song_name"] = str(params["song_name"])
            if cfg["song_select"]["song_name"]:
                cfg["song_select"]["enabled"] = True

        auto_select = params.get("auto_select", False)
        if auto_select:
            cfg.setdefault("song_select", {})["auto_select"] = True

        song_selector = SongSelector(cfg)
        if not song_selector.enabled:
            logger.info("自动选曲未启用，跳过选歌")
            return CustomAction.RunResult(success=True)

        logger.info("开始选歌: %s", song_selector.song_name)

        scene_gate = SceneGate(cfg)
        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)

        max_wait_frames = target_fps * 60
        wait_count = 0

        while wait_count < max_wait_frames:
            if context.tasker.stopping:
                logger.info("tasker 发出停止信号，退出选歌")
                return CustomAction.RunResult(success=False)

            controller.post_screencap().wait()
            frame = controller.cached_image
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue

            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            state, _ = scene_gate.step(frame)

            if state == STATE_SONG_SELECT:
                scroll_fn = lambda sx, sy, sd: (
                    controller.post_swipe(sx, sy, sx, sy + sd * 100, duration=100).wait(),
                    time.sleep(0.1),
                )
                sel_info = song_selector.step(
                    frame, controller, scroll_func=scroll_fn
                )
                sel_state = sel_info.get("state", "")
                if sel_state == "done":
                    logger.info("选歌完成")
                    return CustomAction.RunResult(success=True)
                elif sel_state == "failed":
                    logger.warning("自动选歌失败")
                    return CustomAction.RunResult(success=False)

            wait_count += 1
            time.sleep(frame_interval)

        logger.warning("选歌超时")
        return CustomAction.RunResult(success=False)