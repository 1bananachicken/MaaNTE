from __future__ import annotations

import math
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch

from utils.logger import logger

from ..Common.utils import click_rect, get_image
from .fish_params import load_custom_action_params

STORE_ROI = (26, 81, 418, 585)
BAIT_TEMPLATE = ["Fish/FishStoreGeneralBait.png"]
BAIT_DETAIL_ROI = (1029, 115, 164, 60)
BAIT_NAMES = [
    "万能鱼饵",
    "萬能魚餌",
    r"(?i)Universal\s*Bait",
    "万能釣り餌",
    "만능 미끼",
]


def _candidate_box(context: Context, image, index: int, threshold: float):
    result = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            template=BAIT_TEMPLATE,
            roi=STORE_ROI,
            threshold=[threshold],
            order_by="Score",
            index=index,
            green_mask=True,
        ),
        image,
    )
    if result is None or not result.hit or result.best_result is None:
        return None
    box = getattr(result.best_result, "box", None)
    if not box or len(box) != 4:
        return None
    return tuple(int(value) for value in box)


def _collect_candidates(
    context: Context, image, threshold: float, max_candidates: int
) -> list[tuple[int, int, int, int]]:
    candidates = []
    for index in range(max_candidates):
        box = _candidate_box(context, image, index, threshold)
        if box is None:
            break
        x, y, w, h = box
        center = (x + w / 2, y + h / 2)
        if any(
            math.hypot(center[0] - old_center[0], center[1] - old_center[1]) < 24
            for old_center in (
                (old_x + old_w / 2, old_y + old_h / 2)
                for old_x, old_y, old_w, old_h in candidates
            )
        ):
            continue
        candidates.append(box)
    return candidates


def _is_general_bait(context: Context, image) -> bool:
    result = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(roi=BAIT_DETAIL_ROI, expected=BAIT_NAMES, threshold=0.3),
        image,
    )
    return bool(result and result.hit)


@AgentServer.custom_action("auto_choose_fish_store_general_bait")
class AutoChooseFishStoreGeneralBait(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = load_custom_action_params(argv.custom_action_param)
        threshold = float(params.get("candidate_threshold", 0.6))
        max_candidates = max(1, int(params.get("max_candidates", 8)))
        detail_timeout_ms = max(200, int(params.get("detail_timeout_ms", 1200)))

        controller = context.tasker.controller
        image = get_image(controller)
        candidates = _collect_candidates(context, image, threshold, max_candidates)
        if not candidates:
            logger.warning("商店中未找到万能鱼饵候选")
            return CustomAction.RunResult(success=False)

        logger.debug("商店找到 %d 个鱼饵候选", len(candidates))
        for index, box in enumerate(candidates, start=1):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            click_rect(controller, box)
            deadline = time.monotonic() + detail_timeout_ms / 1000.0
            while time.monotonic() < deadline:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                image = get_image(controller)
                if _is_general_bait(context, image):
                    logger.debug("第 %d 个候选通过万能鱼饵名称确认", index)
                    return CustomAction.RunResult(success=True)
                time.sleep(0.05)
            logger.debug("第 %d 个候选未通过名称确认，尝试下一个", index)

        logger.warning("所有鱼饵候选均未通过万能鱼饵名称确认")
        return CustomAction.RunResult(success=False)
