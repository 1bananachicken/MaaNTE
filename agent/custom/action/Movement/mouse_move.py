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

        dx = int(params.get("dx", 0))
        dy = int(params.get("dy", 0))
        steps = params.get("steps", 30)
        step_delay = params.get("step_delay", 0.03)
        align = params.get("align", True)

        if steps < 1:
            steps = 1

        step_x = dx / steps
        step_y = dy / steps

        moved_x = 0
        moved_y = 0
        for i in range(steps):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            target_x = int(round(step_x * (i + 1)))
            target_y = int(round(step_y * (i + 1)))
            delta_x = target_x - moved_x
            delta_y = target_y - moved_y
            if delta_x != 0 or delta_y != 0:
                controller.post_relative_move(delta_x, delta_y).wait()
            moved_x = target_x
            moved_y = target_y
            time.sleep(step_delay)

        remainder_x = dx - moved_x
        remainder_y = dy - moved_y
        if remainder_x != 0 or remainder_y != 0:
            controller.post_relative_move(remainder_x, remainder_y).wait()

        if align and (dx != 0 or dy != 0):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            controller.post_key_down(_KEY_W).wait()
            time.sleep(_ALIGN_DURATION)
            controller.post_key_up(_KEY_W).wait()
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)
