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
        repeat = False
        repeat_count = 1
        if argv.custom_action_param:
            try:
                print(f"custom_action_param type: {type(argv.custom_action_param)}")
                print(f"custom_action_param value: {argv.custom_action_param}")


                if isinstance(argv.custom_action_param, str):
                    params = json.loads(argv.custom_action_param)
                elif isinstance(argv.custom_action_param, dict):
                    params = argv.custom_action_param
                else:
                    params = {}

                mode = params.get("mode", "single")
                repeat = params.get("repeat", False)
                repeat_count = int(params.get("repeat_count", 1))
                if repeat_count < 1:
                    repeat_count = 1
            except Exception:
                pass

        player = TetrisGamePlayer()

        if not repeat:
            success = player.run(controller, tasker, mode=mode)
            return CustomAction.RunResult(success=success)

        initial_scene = player._detect_initial_scene(controller, tasker)
        if initial_scene is None:
            print("Cannot detect Tetris initial scene, ending task.")
            return CustomAction.RunResult(success=False)

        scene_name = initial_scene["name"]
        print(f"Auto-repeat: initial scene={scene_name}, repeat_count={repeat_count}")

        if scene_name == "world_no_prompt":
            print(
                "World scene detected but no Tetris entrance prompt found, ending task."
            )
            return CustomAction.RunResult(success=False)

        if scene_name == "unknown":
            print("Unknown initial scene, ending task.")
            return CustomAction.RunResult(success=False)

        if scene_name == "result":
            print("Starting at result screen, dismissing...")
            player._press_escape(controller)
            if not player._sleep_with_stop(tasker, 1.0):
                return CustomAction.RunResult(success=False)
            if not player._back_to_world_from_anywhere(controller, tasker):
                print("Failed to dismiss result screen.")
                return CustomAction.RunResult(success=False)
            scene_name = "world_prompt"

        player.mode = mode

        started_in_game = scene_name in ("game_active", "game_idle")

        if started_in_game:
            print("Starting in game, playing current round (not counted)...")
            success = player.run(controller, tasker, mode=mode)
            if not success:
                return CustomAction.RunResult(success=False)
            if tasker.stopping:
                return CustomAction.RunResult(success=False)
            if not player._back_to_world_from_anywhere(controller, tasker):
                print("Failed to return to world after initial play.")
                return CustomAction.RunResult(success=False)

        for i in range(repeat_count):
            if tasker.stopping:
                return CustomAction.RunResult(success=False)

            print(f"=== Auto-repeat round {i + 1}/{repeat_count} ===")
            player.reset()

            success = player._navigate_to_game_and_play(controller, tasker)
            if not success:
                print(f"Round {i + 1} failed.")
                return CustomAction.RunResult(success=False)

            if tasker.stopping:
                return CustomAction.RunResult(success=False)

            if i < repeat_count - 1:
                if not player._back_to_world_from_anywhere(controller, tasker):
                    print(f"Failed to return to world after round {i + 1}.")
                    return CustomAction.RunResult(success=False)

        print(f"=== All {repeat_count} rounds completed ===")
        return CustomAction.RunResult(success=True)
