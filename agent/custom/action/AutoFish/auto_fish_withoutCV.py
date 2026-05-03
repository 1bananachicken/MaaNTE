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
        max_try_item = 10
        print("[auto_fish_without_cv] 开始：等待鱼上钩")
        # 等待鱼上钩
        while not context.tasker.stopping:
            image = context.tasker.controller.post_screencap().wait().get()
            fish_hooked = context.run_recognition("FishHooked", image)
            time.sleep(0.1)
            if fish_hooked and fish_hooked.hit:
                print("[auto_fish_without_cv] 识别到鱼上钩，执行 FishHook")
                context.run_action("FishHook")
                break

        print("[auto_fish_without_cv] 进入控条阶段（绿条/光标对齐）")
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
                max_try_item -= 1
                print(
                    f"[auto_fish_without_cv] 识别不完整（绿条或光标未命中），"
                    f"剩余尝试次数: {max_try_item}"
                )
                if max_try_item <= 0:
                    print("[auto_fish_without_cv] 尝试次数用尽，控条失败")
                    return CustomAction.RunResult(success=False)
                continue

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
                print(
                    f"[auto_fish_without_cv] 控条: offset={offset:.1f}px, "
                    f"条内={cursor_in_bar}, 模式={mode}, 时长={duration_ms}ms → FishLeft"
                )
                context.run_action(
                    "FishLeft",
                    pipeline_override={
                        "FishLeft": {"action": {"param": param_override}},
                    },
                )
            elif offset < -deadzone:
                print(
                    f"[auto_fish_without_cv] 控条: offset={offset:.1f}px, "
                    f"条内={cursor_in_bar}, 模式={mode}, 时长={duration_ms}ms → FishRight"
                )
                context.run_action(
                    "FishRight",
                    pipeline_override={
                        "FishRight": {"action": {"param": param_override}},
                    },
                )

        print("[auto_fish_without_cv] 任务结束（success=True）")
        return CustomAction.RunResult(success=True)
