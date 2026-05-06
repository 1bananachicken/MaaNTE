import json
import logging
import re
import time
from typing import Any

import cv2

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JOCR

from ..utils.config import load_rhythm_config


logger = logging.getLogger(__name__)

_DEFAULT_ROI = (544, 622, 184, 46)
_DEFAULT_COST_PATTERN = r"(\d+)"
_DEFAULT_MIN_CONFIRM_READS = 2
_DEFAULT_CONFIRM_INTERVAL_SEC = 0.3
_DEFAULT_VITALITY_THRESHOLD = 1

_repeat_index: int = 0
_vitality_cost: int | None = None


def _detect_cost_vitality(context: Context, frame: Any, cfg: dict[str, Any]) -> int:
    if frame is None or frame.size == 0:
        return 0

    vcfg = cfg.get("vitality_detect") or {}
    roi_list = vcfg.get("roi", list(_DEFAULT_ROI))
    roi = (
        tuple(roi_list)
        if isinstance(roi_list, list) and len(roi_list) == 4
        else _DEFAULT_ROI
    )
    cost_pattern_str = str(vcfg.get("cost_pattern", _DEFAULT_COST_PATTERN))
    min_confirm = max(1, int(vcfg.get("min_confirm_reads", _DEFAULT_MIN_CONFIRM_READS)))
    confirm_interval = float(
        vcfg.get("confirm_interval_sec", _DEFAULT_CONFIRM_INTERVAL_SEC)
    )

    try:
        cost_pattern = re.compile(cost_pattern_str)
    except re.error:
        logger.warning("活力检测正则无效: %s，使用默认", cost_pattern_str)
        cost_pattern = re.compile(_DEFAULT_COST_PATTERN)

    confirmed_costs: list[int] = []
    last_text = ""

    for attempt in range(min_confirm + 2):
        if attempt > 0:
            time.sleep(confirm_interval)
            controller = context.tasker.controller
            controller.post_screencap().wait()
            frame = controller.cached_image
            if frame is None or frame.size == 0:
                continue
            if len(frame.shape) == 3 and frame.shape[2] == 4:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        detail = context.run_recognition_direct(
            JRecognitionType.OCR, JOCR(roi=roi), frame
        )
        if detail is None or not detail.hit or detail.best_result is None:
            logger.debug(
                "消耗活力 OCR 未命中 (尝试 %d/%d)", attempt + 1, min_confirm + 2
            )
            continue

        text = (
            detail.best_result.text
            if hasattr(detail.best_result, "text")
            else str(detail.best_result)
        )
        if text:
            last_text = text
        m = cost_pattern.search(text)
        if m:
            cost = int(m.group(1))
            logger.debug(
                "消耗活力 OCR: %s -> %d (尝试 %d/%d)",
                text,
                cost,
                attempt + 1,
                min_confirm + 2,
            )
            confirmed_costs.append(cost)
            if len(confirmed_costs) >= min_confirm:
                break
        else:
            logger.debug(
                "消耗活力 OCR 未匹配: %s (尝试 %d/%d)",
                text,
                attempt + 1,
                min_confirm + 2,
            )

    if confirmed_costs:
        from collections import Counter

        most_common_cost, count = Counter(confirmed_costs).most_common(1)[0]
        logger.info(
            "活力消耗确认: %d (出现 %d/%d 次, 原始: %s)",
            most_common_cost,
            count,
            len(confirmed_costs),
            last_text,
        )
        return most_common_cost

    if last_text:
        logger.warning("活力检测: 多次 OCR 均未能匹配消耗值，最后文本: %s", last_text)
    else:
        logger.warning("活力检测: 所有 OCR 尝试均未命中，可能活力耗尽或区域错误")

    return 0


@AgentServer.custom_action("auto_rhythm_vitality_on_results")
class AutoRhythmVitalityOnResults(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _vitality_cost

        cfg = load_rhythm_config()
        vcfg = cfg.get("vitality_detect") or {}
        vitality_enabled = bool(vcfg.get("enabled", True))

        if not vitality_enabled:
            logger.debug("活力检测已禁用，跳过结算页 OCR")
            _vitality_cost = None
            return CustomAction.RunResult(success=True)

        controller = context.tasker.controller
        controller.post_screencap().wait()
        frame = controller.cached_image
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cost = _detect_cost_vitality(context, frame, cfg)
        _vitality_cost = cost
        logger.info("结算页活力消耗: %d", cost)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("auto_rhythm_repeat_decision")
class AutoRhythmRepeatDecision(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _repeat_index, _vitality_cost

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
            vcfg = cfg.get("vitality_detect") or {}
            vitality_threshold = int(
                vcfg.get("vitality_threshold", _DEFAULT_VITALITY_THRESHOLD)
            )

            cost = _vitality_cost
            if cost is None:
                logger.warning("未获取结算页活力消耗值，尝试现场识别")
                controller = context.tasker.controller
                controller.post_screencap().wait()
                frame = controller.cached_image
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                cost = _detect_cost_vitality(context, frame, cfg)

            if cost < vitality_threshold:
                logger.info(
                    "活力已耗尽 (消耗=%d < 阈值=%d)，停止连打",
                    cost,
                    vitality_threshold,
                )
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
            _vitality_cost = None
            context.override_next("RhythmRepeatCheck", ["RhythmExit"])
        else:
            context.override_next("RhythmRepeatCheck", ["RhythmSelectSong"])

        return CustomAction.RunResult(success=True)
