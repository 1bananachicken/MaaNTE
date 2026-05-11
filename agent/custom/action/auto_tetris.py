import json
import re
import time

import cv2
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

from .Tetris.feats.play import TetrisGamePlayer

_round_count = 0
_target_round = 0
_single_shot_done = False
_allow_speed_drop = False


@AgentServer.custom_action("tetris_reset_context")
class TetrisResetContext(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _target_round, _single_shot_done, _allow_speed_drop
        _round_count = 0
        _target_round = 0
        _single_shot_done = False
        _allow_speed_drop = False

        params = (
            json.loads(argv.custom_action_param)
            if isinstance(argv.custom_action_param, str)
            else (argv.custom_action_param or {})
        )
        _allow_speed_drop = params.get("allow_speed_drop", False)
        print("[AutoTetris] Task stats reset.")
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("auto_tetris")
class AutoTetris(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _target_round, _single_shot_done

        controller = context.tasker.controller
        tasker = context.tasker

        params = (
            json.loads(argv.custom_action_param)
            if isinstance(argv.custom_action_param, str)
            else (argv.custom_action_param or {})
        )

        mode = params.get("mode", "single")
        use_all_vitality = params.get("use_all_vitality", False)
        allow_speed_drop = params.get("allow_speed_drop", _allow_speed_drop)
        rc = params.get("repeat_count", 1)
        try:
            new_target = int(rc) if rc else 0
        except (ValueError, TypeError):
            print(f"[AutoTetris] repeat_count parse failed: {rc}")
            new_target = 0

        if new_target > 0 and _target_round != new_target:
            _round_count = 0
            _target_round = new_target
            _single_shot_done = False

        if not use_all_vitality and _single_shot_done:
            print(f"[AutoTetris] All {_target_round} rounds already done. Stopping task.")
            tasker.post_stop()  # Stop the task after finishing the target rounds
            controller.post_key_down(27) # Press ESC to exit game screen
            time.sleep(0.05)
            controller.post_key_up(27)
            return CustomAction.RunResult(success=False)

        player = TetrisGamePlayer()
        player.context = context
        player.mode = mode
        player.fast_drop = allow_speed_drop
        player.debug = params.get("debug", False)
        success = player.play_round(controller, tasker)

        if not success:
            _round_count = 0
            return CustomAction.RunResult(success=False)

        if not use_all_vitality:
            _round_count += 1
            print(f"[AutoTetris] Finished round {_round_count}/{_target_round}")

            if _round_count >= _target_round:
                _single_shot_done = True
                print("[AutoTetris] All rounds finished.")
                tasker.post_stop()  # Stop the task after finishing the target rounds
                controller.post_key_down(27) # Press ESC to exit game screen
                time.sleep(0.05)
                controller.post_key_up(27)
                return CustomAction.RunResult(success=True)

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("tetris_check_vitality_action")
class TetrisCheckVitalityAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        roi = [451, 290, 371, 20]
        controller = context.tasker.controller
        controller.post_screencap().wait()
        frame = controller.cached_image

        vitality = 0
        if frame is not None and frame.size > 0:
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            detail = context.run_recognition_direct(
                JRecognitionType.OCR, JOCR(roi=roi), frame
            )

            if detail is not None and detail.hit and detail.all_results:
                for r in detail.all_results:
                    t = r.text if hasattr(r, "text") else str(r)
                    numbers = re.findall(r"\d+", t)
                    if numbers:
                        vitality = int(numbers[-1])
                print(f"[TetrisCheckVitality] vitality={vitality}")
            else:
                print("[TetrisCheckVitality] OCR no hit")
        else:
            print("[TetrisCheckVitality] screencap failed")

        if vitality == 0:
            print("[TetrisCheckVitality] vitality == 0, stopping")
            controller.post_key_down(27)
            time.sleep(0.05)
            controller.post_key_up(27)
            return CustomAction.RunResult(success=False)

        if vitality < 0:
            print("[TetrisCheckVitality] OCR failed or vitality not found, assuming vitality available")

        controller.post_key_down(27)
        time.sleep(0.05)
        controller.post_key_up(27)
        return CustomAction.RunResult(success=True)
