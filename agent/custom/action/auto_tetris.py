import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .Tetris.feats.play import TetrisGamePlayer


@AgentServer.custom_action("auto_tetris")
class AutoTetris(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        tasker = context.tasker

        mode = "single"
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                mode = params.get("mode", "single")
            except Exception:
                pass

        player = TetrisGamePlayer()
        success = player.run(controller, tasker, mode=mode)

        return CustomAction.RunResult(success=success)
