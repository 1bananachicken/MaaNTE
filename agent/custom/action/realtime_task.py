import json
from typing import List

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


HOLDER_NODE_NAME = "__RealTimeTaskAction_Holder"


def _parse_nodes(custom_action_param: str) -> List[str]:
    if not custom_action_param:
        raise ValueError("RealTimeTaskAction: empty custom_action_param")

    params = json.loads(custom_action_param)
    if not isinstance(params, dict):
        raise ValueError(f"RealTimeTaskAction: invalid JSON object: {custom_action_param}")

    nodes = params.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError(f"RealTimeTaskAction: 'nodes' missing, not an array, or empty: {custom_action_param}")

    for v in nodes:
        if not isinstance(v, str):
            raise ValueError("RealTimeTaskAction: every entry in 'nodes' must be a string")

    return nodes


def _build_pipeline_override(nodes: List[str]) -> dict:
    return {HOLDER_NODE_NAME: {"next": nodes}}


@AgentServer.custom_action("RealTimeTaskAction")
class RealTimeTaskAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        nodes = _parse_nodes(argv.custom_action_param)
        pipeline_override = _build_pipeline_override(nodes)

        while not context.tasker.stopping:
            result = context.run_task(HOLDER_NODE_NAME, pipeline_override)
            if result is None:
                print("RealTimeTaskAction: RunTask returned None, continue loop")

        print("RealTimeTaskAction: tasker stopping signal received, exit loop")
        return CustomAction.RunResult(success=True)
