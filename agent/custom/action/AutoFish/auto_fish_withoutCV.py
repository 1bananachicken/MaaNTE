from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

import time
import json

from ..Common.logger import get_logger

logger = get_logger("auto_fish_without_cv")

# 长按左/右键时，光标在进度条上水平移动约 200 像素/秒，用于将偏移（像素）换算为 LongPress 时长
CURSOR_PX_PER_SEC = 200.0


@AgentServer.custom_action("auto_fish_without_cv")
class AutoFishWithoutCV(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:

        deadzone = 10  # 光标与绿条中心的距离在 deadzone（像素）以内时不操作，避免过度频繁地轻微调整导致的抖动
        max_try_item = 5  # 识别不完整（绿条或光标未命中）的最大尝试次数，超过后放弃本次钓鱼，重新抛竿（执行 FishHook）
        factor = 1.5  # 控条时长的调整因子，实际时长 = 基础时长 * factor，基础时长 = (光标与绿条中心的像素偏移 / CURSOR_PX_PER_SEC) * 1000ms，增加 factor 可以适当补偿识别误差和按键响应延迟
        cap_ms = (
            1500  # 控条时长的上限（毫秒），避免因识别到较大偏移时按键过久，导致过度补偿
        )
        floor_ms = (
            50  # 控条时长的下限（毫秒），避免因识别到较小偏移时按键过短，导致补偿不足
        )
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                deadzone = params.get("deadzone", deadzone)
                max_try_item = params.get("max_try", max_try_item)
                factor = params.get("factor", factor)
                cap_ms = params.get("cap_ms", cap_ms)
                floor_ms = params.get("floor_ms", floor_ms)
            except Exception:
                pass

        logger.info("钓鱼开始：进入控条阶段（绿条/光标对齐）")
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
                logger.warning(
                    f"识别不完整（绿条或光标未命中），剩余尝试次数: {max_try_item}"
                )

                if max_try_item <= 0:
                    logger.error("尝试次数用尽，控条失败")
                    return CustomAction.RunResult(success=True)
                continue

            green_bar_x, green_bar_y, green_bar_w, green_bar_h = green_bar.box
            cursor_x, cursor_y, cursor_w, cursor_h = cursor.box

            green_bar_center_x = green_bar_x + green_bar_w / 2
            cursor_center_x = cursor_x + cursor_w / 2

            # 与 auto_fish.py 一致：offset = 滑块 x - 目标中心 x（此处用识别框中心对应 slider / target）
            offset = cursor_center_x - green_bar_center_x

            abs_offset = abs(offset)
            base_ms = (abs_offset / CURSOR_PX_PER_SEC) * 1000.0
            duration_ms = min(cap_ms, max(floor_ms, int(base_ms * factor)))

            # 键码与 LongPressKey 定义见资源 pipeline FishKey（FishLeft / FishRight），此处只覆盖时长
            param_override = {"duration": duration_ms}

            if offset > deadzone:
                logger.info(
                    f"控条: offset={offset:.1f}px, 时长={duration_ms}ms → FishLeft"
                )
                context.run_action(
                    "FishLeft",
                    pipeline_override={
                        "FishLeft": {"action": {"param": param_override}},
                    },
                )
            elif offset < -deadzone:
                logger.info(
                    f"控条: offset={offset:.1f}px, 时长={duration_ms}ms → FishRight"
                )
                context.run_action(
                    "FishRight",
                    pipeline_override={
                        "FishRight": {"action": {"param": param_override}},
                    },
                )

        logger.info("任务结束（success=True）")
        return CustomAction.RunResult(success=True)
