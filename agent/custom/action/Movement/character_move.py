import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

_KEY_MAP = {"W": 87, "A": 65, "S": 83, "D": 68}


@AgentServer.custom_action("character_move")
class CharacterMoveAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        key = params.get("key", None)
        try:
            duration = float(params.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0

        vk = _KEY_MAP.get(str(key).upper()) if key else None
        if vk and duration > 0:
            controller.post_key_down(vk).wait()
            deadline = time.time() + duration
            while time.time() < deadline:
                if context.tasker.stopping:
                    controller.post_key_up(vk).wait()
                    return CustomAction.RunResult(success=False)
                time.sleep(0.05)
            controller.post_key_up(vk).wait()

        return CustomAction.RunResult(success=True)
