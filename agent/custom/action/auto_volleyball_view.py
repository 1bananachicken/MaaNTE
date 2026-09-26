import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger


@AgentServer.custom_action("volleyball_view_setup")
class VolleyballViewSetup(CustomAction):
    """一次性视角初始化：鼠标下移到俯视角度（Z视角切换有记忆，仅重登后需手动切）。"""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        dy = 200
        if argv.custom_action_param:
            import json
            try:
                p = json.loads(argv.custom_action_param) if isinstance(argv.custom_action_param, str) else argv.custom_action_param
                dy = int(p.get("dy", dy))
            except Exception:
                pass

        try:
            controller.post_relative_move(0, dy).wait()
            time.sleep(0.5)
            return CustomAction.RunResult(success=True)
        except Exception:
            logger.exception("VolleyballViewSetup: failed")
            return CustomAction.RunResult(success=False)
