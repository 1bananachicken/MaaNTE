import json

from .utils import click_rect, click_rect_720

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


@AgentServer.custom_action("click_override")
class ClickOverride(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller

        if argv.custom_action_param is not None:
            try:
                params = json.loads(argv.custom_action_param)
                target = params.get("target")
                if not target or len(target) != 4:
                    logger.warning(f"click_override invalid rect parameter: {argv.custom_action_param}")
                    return CustomAction.RunResult(success=False)
            except Exception as e:
                logger.warning(f"click_override parse parameter failed: {e}")
                return CustomAction.RunResult(success=False)

            click_rect_720(controller, target, 0.005)
            logger.debug(f"click_override clicked baseline rect: {target}")
            return CustomAction.RunResult(success=True)

        elif argv.reco_detail is not None:
            click_rect(controller, argv.box, 0.005)
            return CustomAction.RunResult(success=True)

        else:
            logger.warning("click_override missing parameters")
            return CustomAction.RunResult(success=False)
