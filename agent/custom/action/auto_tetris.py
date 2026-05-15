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
_task_started = False
_task_config = {}


@AgentServer.custom_action("tetris_reset_context")
class TetrisResetContext(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _target_round, _single_shot_done, _allow_speed_drop
        global _task_started, _task_config
        _round_count = 0
        _target_round = 0
        _single_shot_done = False
        _allow_speed_drop = False
        _task_started = False
        _task_config = {}

        params = (
            json.loads(argv.custom_action_param)
            if isinstance(argv.custom_action_param, str)
            else (argv.custom_action_param or {})
        )
        _allow_speed_drop = params.get("allow_speed_drop", False)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("auto_tetris")
class AutoTetris(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _target_round, _single_shot_done
        global _task_started, _task_config

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
            new_target = 0

        if new_target > 0 and _target_round != new_target:
            _round_count = 0
            _target_round = new_target
            _single_shot_done = False

        if not _task_started:
            _task_started = True
            _task_config = {
                "mode": mode,
                "use_all_vitality": use_all_vitality,
                "allow_speed_drop": allow_speed_drop,
                "target_round": new_target if new_target > 0 else 1,
            }
            mode_label = "单人" if mode == "single" else "多人"
            parts = [f"模式: {mode_label}"]
            if new_target > 1:
                parts.append(f"连打: {new_target}轮")
            if use_all_vitality:
                parts.append("消耗所有活力: 是")
            if allow_speed_drop:
                parts.append("允许速降: 是")
            print(f"[泯除方块] 任务开始 | {' | '.join(parts)}")

        if not use_all_vitality and _single_shot_done:
            tasker.post_stop()
            controller.post_key_down(27)
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
            print("[泯除方块] 任务异常结束")
            return CustomAction.RunResult(success=False)

        if not use_all_vitality:
            _round_count += 1
            print(f"[泯除方块] 进度: {_round_count}/{_target_round}")

            if _round_count >= _target_round:
                _single_shot_done = True
                print("[泯除方块] 任务完成")
                tasker.post_stop()
                controller.post_key_down(27)
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

        if vitality == 0:
            print("[泯除方块] 活力耗尽，任务完成")
            controller.post_key_down(27)
            time.sleep(0.05)
            controller.post_key_up(27)
            return CustomAction.RunResult(success=False)

        controller.post_key_down(27)
        time.sleep(0.05)
        controller.post_key_up(27)
        return CustomAction.RunResult(success=True)
