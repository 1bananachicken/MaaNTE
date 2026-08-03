import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


def load_params(custom_action_param) -> dict:
    """兼容 None / dict / JSON 字符串（含 "null"、"{}"）的 custom_action_param 解析。"""
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    try:
        params = json.loads(custom_action_param)
    except Exception:
        return {}
    return params if isinstance(params, dict) else {}


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
            return CustomAction.RunResult(success=False)
        context.override_pipeline({target: {"enabled": True}})
        return CustomAction.RunResult(success=True)
