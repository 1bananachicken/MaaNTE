import json

from .utils import click_rect

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

@AgentServer.custom_action("click_override")
class ClickOverride(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Click Action Started ===")
        controller = context.tasker.controller

        params = load_params(argv.custom_action_param)
        target = params.get("target")

        if target and len(target) == 4:
            click_rect(controller, target, 0.005)
            print(f"Clicked at rect: {target}")
            return CustomAction.RunResult(success=True)

        if argv.reco_detail is not None:
            click_rect(controller, argv.box, 0.005)
            print(f"Clicked at reco box: {argv.box}")
            return CustomAction.RunResult(success=True)

        print("No valid parameters provided for click action.")
        return CustomAction.RunResult(success=False)
