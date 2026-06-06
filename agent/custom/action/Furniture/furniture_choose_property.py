import json
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from ..Common.utils import click_rect, get_image, swipe_720
from utils.logger import logger

_TOP_RESET_COUNT = 1
_MAX_SCROLL_COUNT = 1
_SWIPE_DURATION_MS = 500
_SWIPE_DELAY_SEC = 0.1
_TOP_SWIPE_BEGIN = (40, 170)
_TOP_SWIPE_END = (160, 470)
_NEXT_SWIPE_BEGIN = (40, 470)
_NEXT_SWIPE_END = (160, 170)


def _screencap(controller):
    return get_image(controller)


def _box_to_rect(box):
    if isinstance(box, (list, tuple)):
        return list(box)
    return [box.x, box.y, box.w, box.h]


def _click_rect(controller, rect):
    click_rect(controller, rect, delay=0.05, move=True)
    time.sleep(0.05)


def _swipe(controller, begin, end):
    swipe_720(controller, begin, end, duration=_SWIPE_DURATION_MS)
    time.sleep(_SWIPE_DELAY_SEC)


def _parse_target(custom_action_param) -> str | None:
    if not custom_action_param:
        return None

    if isinstance(custom_action_param, str):
        try:
            params = json.loads(custom_action_param)
        except json.JSONDecodeError:
            return custom_action_param.strip() or None

        if isinstance(params, str):
            return params.strip() or None
        if isinstance(params, dict):
            target = params.get("target")
            if isinstance(target, str):
                return target.strip() or None
        return None

    if isinstance(custom_action_param, dict):
        target = custom_action_param.get("target")
        if isinstance(target, str):
            return target.strip() or None

    return None


@AgentServer.custom_action("furniture_choose_property")
class FurnitureChooseProperty(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        target = _parse_target(getattr(argv, "custom_action_param", None))
        if not target:
            logger.warning(f"furniture_choose_property missing target:{argv}")
            return CustomAction.RunResult(success=False)

        controller = context.tasker.controller

        for _ in range(_TOP_RESET_COUNT):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            _swipe(controller, _TOP_SWIPE_BEGIN, _TOP_SWIPE_END)

        for scroll_index in range(_MAX_SCROLL_COUNT + 1):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            image = _screencap(controller)
            result = context.run_recognition(target, image)
            if result and result.hit and result.box is not None:
                rect = _box_to_rect(result.box)
                logger.debug(
                    f"furniture_choose_property target found: {target} rect={rect}"
                )
                _click_rect(controller, rect)
                return CustomAction.RunResult(success=True)

            if scroll_index >= _MAX_SCROLL_COUNT:
                break

            logger.debug(
                f"furniture_choose_property target not found, scroll next: {target} step={scroll_index+1}"
            )
            _swipe(controller, _NEXT_SWIPE_BEGIN, _NEXT_SWIPE_END)

        logger.warning(f"furniture_choose_property target not found: {target}")
        return CustomAction.RunResult(success=False)
