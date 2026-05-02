import json
import logging
import os
import time
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from ..utils.presence import (
    STATE_RESULTS,
    SceneGate,
)


logger = logging.getLogger(__name__)

_VK_ESCAPE = 0x1B


def _load_rhythm_config() -> dict[str, Any]:
    from ...auto_rhythm import _load_rhythm_config as _load
    return _load()


@AgentServer.custom_action("auto_rhythm_repeat")
class AutoRhythmRepeat(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        cfg = _load_rhythm_config()

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        if "auto_repeat_count" in params:
            count = int(params["auto_repeat_count"])
            cfg.setdefault("auto_repeat", {})["count"] = count
            if count > 0:
                cfg["auto_repeat"]["enabled"] = True
            else:
                cfg["auto_repeat"]["enabled"] = False

        auto_repeat_cfg = cfg.get("auto_repeat") or {}
        auto_repeat_enabled = bool(auto_repeat_cfg.get("enabled", False))
        auto_repeat_count = int(auto_repeat_cfg.get("count", 1))
        auto_repeat_dismiss_delay = float(auto_repeat_cfg.get("dismiss_delay_sec", 0.8))

        if not auto_repeat_enabled:
            logger.info("自动连打未启用，仅执行一次")
            return CustomAction.RunResult(success=True)

        logger.info("自动连打已启用，目标次数: %d", auto_repeat_count)

        scene_gate = SceneGate(cfg)
        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)

        repeat_index = 0
        results_seen = False
        esc_sent_for_results = False

        max_wait_frames = target_fps * 300
        wait_count = 0

        while wait_count < max_wait_frames:
            if context.tasker.stopping:
                logger.info("tasker 发出停止信号，退出连打")
                return CustomAction.RunResult(success=False)

            controller.post_screencap().wait()
            frame = controller.cached_image
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue

            import cv2
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            state, _ = scene_gate.step(frame)

            if state == STATE_RESULTS:
                if not results_seen:
                    results_seen = True
                    repeat_index += 1
                    esc_sent_for_results = False
                    logger.info(
                        "检测到结算界面: 第 %d/%d 次",
                        repeat_index,
                        auto_repeat_count,
                    )

                if repeat_index >= auto_repeat_count:
                    logger.info("已达到连打次数上限 (%d)，停止", auto_repeat_count)
                    time.sleep(auto_repeat_dismiss_delay)
                    return CustomAction.RunResult(success=True)

                if not esc_sent_for_results:
                    time.sleep(auto_repeat_dismiss_delay)
                    logger.info(
                        "发送 ESC 退出结算界面 (第 %d/%d 次)",
                        repeat_index,
                        auto_repeat_count,
                    )
                    controller.post_click_key(_VK_ESCAPE).wait()
                    esc_sent_for_results = True
                time.sleep(1.0)
                continue

            wait_count += 1
            time.sleep(frame_interval)

        logger.warning("连打等待超时")
        return CustomAction.RunResult(success=False)
