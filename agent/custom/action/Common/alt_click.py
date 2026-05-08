from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger


@AgentServer.custom_action("alt_click")
class AltClick(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:

        if argv.box is None:
            logger.error("No target box from recognition")
            return CustomAction.RunResult(success=False)

        box = [argv.box.x, argv.box.y, argv.box.w, argv.box.h]
        context.run_action("__AltClickAltKeyDownAction")
        context.run_action(
            "__AltClickMouseClickAction",
            pipeline_override={
                "__AltClickMouseClickAction": {"action": {"param": {"target": box}}},
            },
        )
        context.run_action("__AltClickAltKeyUpAction")

        logger.debug(f"Alt+Click at box={argv.box}")
        return CustomAction.RunResult(success=True)
