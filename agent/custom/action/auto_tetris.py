import json
import re
import time

import cv2
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

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
                if isinstance(argv.custom_action_param, str):
                    params = json.loads(argv.custom_action_param)
                elif isinstance(argv.custom_action_param, dict):
                    params = argv.custom_action_param
                else:
                    params = {}

                mode = params.get("mode", "single")
            except Exception:
                pass

        player = TetrisGamePlayer()
        player.context = context
        player.mode = mode
        success = player.play_round(controller, tasker)
        return CustomAction.RunResult(success=success)


@AgentServer.custom_action("tetris_check_vitality_action")
class TetrisCheckVitalityAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        roi = [451, 290, 371, 20]
        controller = context.tasker.controller
        controller.post_screencap().wait()
        frame = controller.cached_image

        vitality = -1
        if frame is not None and frame.size > 0:
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            detail = context.run_recognition_direct(
                JRecognitionType.OCR, JOCR(roi=roi), frame
            )

            if detail is not None and detail.hit and detail.all_results:
                texts = []
                for r in detail.all_results:
                    t = r.text if hasattr(r, "text") else str(r)
                    texts.append(t)
                    numbers = re.findall(r"\d+", t)
                    if numbers:
                        vitality = int(numbers[-1])
                print(f"[TetrisCheckVitality] OCR results={texts} -> vitality={vitality}")
            else:
                print("[TetrisCheckVitality] OCR no hit")
        else:
            print("[TetrisCheckVitality] screencap failed")

        if vitality <= 0:
            print("[TetrisCheckVitality] vitality <= 0, stopping")
            controller.post_key_down(27)
            time.sleep(0.05)
            controller.post_key_up(27)
            return CustomAction.RunResult(success=False)

        controller.post_key_down(27)
        time.sleep(0.05)
        controller.post_key_up(27)
        return CustomAction.RunResult(success=True)
