from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

import time


@AgentServer.custom_action("auto_fish_without_cv")
class AutoFishWithoutCV(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:

        deadzone = 15
        # 等待鱼上钩
        while not context.tasker.stopping:
            image = context.tasker.controller.post_screencap().wait().get()
            fish_hooked = context.run_recognition("FishHooked", image)
            if fish_hooked and fish_hooked.hit:
                break
            time.sleep(0.1)

        # 钓鱼阶段
        while not context.tasker.stopping:
            image = context.tasker.controller.post_screencap().wait().get()
            green_bar = context.run_recognition("FishGreenBar", image)
            cursor = context.run_recognition("FishCursor", image)

            if not (
                green_bar
                and green_bar.hit
                and green_bar.box
                is not None  # 这个是为了消除pylance的warning，实际运行时不应该有None的情况
                and cursor
                and cursor.hit
                and cursor.box
                is not None  # 这个是为了消除pylance的warning，实际运行时不应该有None的情况
            ):
                return CustomAction.RunResult(success=True)

            green_bar_x, green_bar_y, green_bar_w, green_bar_h = green_bar.box
            cursor_x, cursor_y, cursor_w, cursor_h = cursor.box

            green_bar_center_x = green_bar_x + green_bar_w / 2
            cursor_center_x = cursor_x + cursor_w / 2

            # 与 auto_fish.py 一致：offset = 滑块 x - 目标中心 x（此处用识别框中心对应 slider / target）
            offset = cursor_center_x - green_bar_center_x

            bar_left = green_bar_x
            bar_right = green_bar_x + green_bar_w
            cursor_in_bar = bar_left <= cursor_center_x <= bar_right

            abs_offset = abs(offset)
            # 条内：积极对齐中心；条外：激进拉回
            if cursor_in_bar:
                scale = 4
                cap_ms = 720
                floor_ms = 90
                mode = "fine"
            else:
                scale = 7
                cap_ms = 900
                floor_ms = 140
                mode = "aggressive"

            duration_ms = min(cap_ms, max(floor_ms, int(abs_offset * scale)))

            # 键码与 LongPressKey 定义见资源 pipeline FishKey（FishLeft / FishRight），此处只覆盖时长
            param_override = {"duration": duration_ms}

            if offset > deadzone:
                context.run_action(
                    "FishLeft",
                    pipeline_override={
                        "FishLeft": {"action": {"param": param_override}},
                    },
                )
            elif offset < -deadzone:
                context.run_action(
                    "FishRight",
                    pipeline_override={
                        "FishRight": {"action": {"param": param_override}},
                    },
                )

        return CustomAction.RunResult(success=True)
