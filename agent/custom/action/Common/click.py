from .utils import click_rect, load_params

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


@AgentServer.custom_action("click_override")
class ClickOverride(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller

        params = load_params(argv.custom_action_param)
        target = params.get("target")

        if isinstance(target, (list, tuple)) and len(target) == 4:
            click_rect(controller, target, 0.005)
            return CustomAction.RunResult(success=True)

        if argv.reco_detail is not None:
            click_rect(controller, argv.box, 0.005)
            return CustomAction.RunResult(success=True)

        logger.warning(
            "ClickOverride: no valid click target. "
            "custom_action_param=%r, target=%r",
            argv.custom_action_param,
            target,
        )
        return CustomAction.RunResult(success=False)
