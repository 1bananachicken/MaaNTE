import json
import logging
import re
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JOCR

from ..utils.config import load_rhythm_config


logger = logging.getLogger(__name__)

_COST_VITALITY_ROI = (540, 594, 187, 35)
_COST_VITALITY_PATTERN = re.compile(r"-\s*(\d+)")

_repeat_index: int = 0


def _detect_cost_vitality(context: Context, frame: Any) -> int:
    if frame is None or frame.size == 0:
        return 0

    detail = context.run_recognition_direct(
        JRecognitionType.OCR, JOCR(roi=_COST_VITALITY_ROI), frame
    )
    if detail is None or not detail.hit or detail.best_result is None:
        logger.debug("消耗活力 OCR 未命中")
        return 0

    text = detail.best_result.text if hasattr(detail.best_result, "text") else str(detail.best_result)
    m = _COST_VITALITY_PATTERN.search(text)
    if m:
        cost = int(m.group(1))
        logger.debug("消耗活力 OCR: %s -> %d", text, cost)
        return cost

    logger.debug("消耗活力 OCR 未匹配: %s", text)
    return 0


@AgentServer.custom_action("auto_rhythm_repeat_decision")
class AutoRhythmRepeatDecision(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _repeat_index

        cfg = load_rhythm_config()

        params: dict[str, Any] = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        auto_repeat_count = int(params.get("auto_repeat_count", 0))
        auto_repeat_max = bool(params.get("auto_repeat_max", False))

        _repeat_index += 1

        logger.info(
            "连打决策: 第 %d 次 | 目标次数=%s | Max=%s",
            _repeat_index,
            auto_repeat_count if auto_repeat_count > 0 else "∞",
            auto_repeat_max,
        )

        should_exit = False

        if auto_repeat_max:
            controller = context.tasker.controller
            controller.post_screencap().wait()
            frame = controller.cached_image
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cost = _detect_cost_vitality(context, frame)
            if cost == 0:
                logger.info("活力耗尽，停止连打")
                should_exit = True
            else:
                logger.info("消耗活力 %d，继续连打", cost)
        elif auto_repeat_count > 0:
            if _repeat_index >= auto_repeat_count:
                logger.info("已达到连打次数上限 (%d)", auto_repeat_count)
                should_exit = True
        else:
            logger.info("自动连打未启用，完成单次演奏后退出")
            should_exit = True

        if should_exit:
            _repeat_index = 0
            context.override_next("RhythmRepeatCheck", ["RhythmExit"])
        else:
            context.override_next("RhythmRepeatCheck", ["[Anchor]RhythmLoopPoint"])

        return CustomAction.RunResult(success=True)