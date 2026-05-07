import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .Tetris.feats.play import TetrisGamePlayer

PREPARE_ONE_CLICK_POINT = (1016, 244)
PREPARE_ONE_MULTI_CLICK_POINT = (885, 418)
PREPARE_TWO_CLICK_POINT = (1121, 674)


@AgentServer.custom_action("auto_tetris")
class AutoTetris(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        tasker = context.tasker

        mode = "single"
        repeat = False
        repeat_count = 1
        use_all_vitality = False
        if argv.custom_action_param:
            try:
                if isinstance(argv.custom_action_param, str):
                    params = json.loads(argv.custom_action_param)
                elif isinstance(argv.custom_action_param, dict):
                    params = argv.custom_action_param
                else:
                    params = {}

                mode = params.get("mode", "single")
                repeat = params.get("repeat", False)
                repeat_count = int(params.get("repeat_count", 1))
                use_all_vitality = params.get("use_all_vitality", False)
                if repeat_count < 1:
                    repeat_count = 1
            except Exception:
                pass

        player = TetrisGamePlayer()
        player.context = context
        player.mode = mode
        success = player.play_round(controller, tasker)
        return CustomAction.RunResult(success=success)


@AgentServer.custom_action("tetris_press_f")
class TetrisPressF(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        controller.post_key_down(70)
        time.sleep(0.05)
        controller.post_key_up(70)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("tetris_click_mode")
class TetrisClickMode(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        mode = "single"
        if argv.custom_action_param:
            try:
                if isinstance(argv.custom_action_param, str):
                    params = json.loads(argv.custom_action_param)
                elif isinstance(argv.custom_action_param, dict):
                    params = argv.custom_action_param
                else:
                    params = {}
                mode = params.get("mode", "single")
            except Exception:
                pass

        if mode == "multiple":
            x, y = PREPARE_ONE_MULTI_CLICK_POINT
        else:
            x, y = PREPARE_ONE_CLICK_POINT

        controller.post_click(x, y).wait()
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("tetris_click_start_match")
class TetrisClickStartMatch(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        x, y = PREPARE_TWO_CLICK_POINT
        controller.post_click(x, y).wait()
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("tetris_press_esc")
class TetrisPressEsc(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        controller.post_key_down(27)
        time.sleep(0.05)
        controller.post_key_up(27)
        return CustomAction.RunResult(success=True)
