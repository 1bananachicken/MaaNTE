"""从大世界进入钓鱼准备界面。
通过 OCR 识别"钓鱼"选项的坐标，
按下 Alt → 光标移到选项 → 松 Alt → 按 F 进入钓鱼准备界面。
这是可以在后台使用的，在大世界钓鱼旁边选择钓鱼选项进入的操作。
"""

import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

from utils.logger import logger

# 钓鱼选项 OCR 搜索区域
OPTION_ROI = [726, 326, 182, 127]
# 多语言匹配
OPTION_TEXTS = ["钓鱼", "釣魚", "Fishing", "釣り"]


@AgentServer.custom_action("enter_fishprepare")
class EnterFishPrepare(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        # ① 截图
        image = controller.post_screencap().wait().get()

        # ② OCR 识别"钓鱼"选项位置
        ocr = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=OPTION_ROI, expected=OPTION_TEXTS),
            image,
        )
        if not ocr or not ocr.hit or not ocr.box:
            logger.debug("enter_fishprepare: 未识别到钓鱼选项")
            return CustomAction.RunResult(success=False)

        x, y, w, h = ocr.box
        logger.debug("enter_fishprepare: 钓鱼选项位置 (%d, %d, %d, %d)", x, y, w, h)

        # ③ 按下 Alt
        context.run_action("__AltClickAltKeyDownAction")
        time.sleep(0.05)

        # ④ Pipeline TouchMove 挪光标（不点击）
        context.run_action(
            "_enter_fishprepare_move",
            pipeline_override={
                "_enter_fishprepare_move": {
                    "action": {"type": "TouchMove", "param": {"target": [x, y, w, h]}},
                },
            },
        )
        time.sleep(0.2)
        logger.debug("enter_fishprepare: 光标移到 (%d, %d, %d, %d)", x, y, w, h)

        # ⑤ 松开 Alt
        context.run_action("__AltClickAltKeyUpAction")
        time.sleep(0.05)

        # ⑥ 按 F 进入
        controller.post_key_down(0x46).wait()
        time.sleep(0.03)
        controller.post_key_up(0x46).wait()

        logger.debug("enter_fishprepare: 完成")
        return CustomAction.RunResult(success=True)
