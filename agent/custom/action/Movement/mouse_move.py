import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

_KEY_W = 87
_ALIGN_DURATION = 0.1


@AgentServer.custom_action("mouse_relative_move")
class MouseMoveAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        steps = params.get("steps", 30)
        step_delay = params.get("step_delay", 0.03)
        align = params.get("align", True)

        if steps < 1:
            steps = 1

        step_x = dx // steps
        step_y = dy // steps

        for _ in range(steps):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            controller.post_relative_move(step_x, step_y).wait()
            time.sleep(step_delay)

        if align and (dx != 0 or dy != 0):
            controller.post_key_down(_KEY_W).wait()
            time.sleep(_ALIGN_DURATION)
            controller.post_key_up(_KEY_W).wait()

        return CustomAction.RunResult(success=True)
