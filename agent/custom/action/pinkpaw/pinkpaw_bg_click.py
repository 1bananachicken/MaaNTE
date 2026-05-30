"""
后台点击工具 - 通过 MAA controller.click() 发送点击
使用 MAA 框架原生接口，不依赖 Windows 消息队列
"""

import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


def bg_click_at(context: Context, x: int, y: int) -> bool:
    """通过 MAA controller.post_click() 点击指定坐标"""
    job = context.tasker.controller.post_click(x, y)
    job.wait()
    return job.succeeded


@AgentServer.custom_action("pinkpaw_bg_click")
class PinkPawBGClick(CustomAction):
    """
    后台点击 custom action
    优先点击 OCR 识别框的中心坐标（argv.box），
    没有识别框时才使用 custom_action_param 里的固定坐标: {"x": 950, "y": 340}
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        x, y = None, None

        # 优先使用 OCR/识别框的中心坐标（Rect: x, y, w, h）
        box = getattr(argv, 'box', None)
        if box and hasattr(box, 'x') and hasattr(box, 'w') and box.w > 0:
            x = int(box.x + box.w / 2)
            y = int(box.y + box.h / 2)

        # 没有识别框时，fallback 到 custom_action_param 里的固定坐标
        if x is None or y is None:
            param = {}
            raw = argv.custom_action_param if hasattr(argv, 'custom_action_param') else None
            if raw:
                try:
                    if isinstance(raw, dict):
                        parsed = raw
                    else:
                        parsed = json.loads(raw)

                    if isinstance(parsed, dict):
                        # 情况3: 外层是完整param对象，x/y 在 custom_action_param 里
                        if "custom_action_param" in parsed:
                            inner = parsed["custom_action_param"]
                            if isinstance(inner, str):
                                inner = json.loads(inner)
                            if isinstance(inner, dict):
                                param = inner
                        # 情况1/2: 直接就是 {"x": ..., "y": ...}
                        elif "x" in parsed or "y" in parsed:
                            param = parsed
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            x = param.get("x", 640)
            y = param.get("y", 360)

        success = bg_click_at(context, x, y)
        return CustomAction.RunResult(success=success)
