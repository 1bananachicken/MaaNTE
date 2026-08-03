from .utils import load_params

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


@AgentServer.custom_action("enable_node")
class EnableNode(CustomAction):
    """动态启用一个 pipeline 节点（等效 pipeline_override 将 enabled 置 true）。

    custom_action_param:
        target: 要启用的节点名，如 "LixiangguanRouteEntrance"
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = load_params(argv.custom_action_param)
        target = params.get("target")
        if not target:
            logger.warning(
                "EnableNode: missing target node name. custom_action_param=%r",
                argv.custom_action_param,
            )
            return CustomAction.RunResult(success=False)
        context.override_pipeline({target: {"enabled": True}})
        return CustomAction.RunResult(success=True)
