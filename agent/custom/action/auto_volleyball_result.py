"""
自动排球 - 结算画面点击
根据 custom_action_param 中的 x/y 坐标点击对应按钮（重新开始或离开）。
"""
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("volleyball_click_result")
class VolleyballClickResult(CustomAction):
    """结算画面点击。

    custom_action_param (JSON):
      x: int    点击横坐标
      y: int    点击纵坐标
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        x = 140
        y = 610
        if argv.custom_action_param:
            try:
                import json
                p = json.loads(argv.custom_action_param)
                x = int(p.get("x", x))
                y = int(p.get("y", y))
            except Exception:
                pass

        controller = context.tasker.controller
        controller.post_click(x, y).wait()
        return CustomAction.RunResult(success=True)
