from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

FURNITURE_LIST = [
    "FurnitureHamsterBall",
    "FurnitureMianmian",
    "FurnitureWoodenBox",
]


@AgentServer.custom_action("furniture_claim")
class FurnitureClaim(CustomAction):
    def run(
        self, context: Context, _argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        claimed_count = 0
        for name in FURNITURE_LIST:
            image = controller.post_screencap().wait().get()
            result = context.run_recognition(name, image)
            if result and result.box:
                roi = [result.box.x, result.box.y, result.box.w, result.box.h]
                result = context.run_recognition(
                    "FurnitureClaim",
                    image,
                    pipeline_override={
                        "FurnitureClaim": {"recogniton": {"param": roi}}
                    },
                )
                if result.hit:
                    context.run_task(
                        "FurnitureClaim",
                        pipeline_override={
                            "FurnitureClaim": {"recogniton": {"param": roi}}
                        },
                    )
                    logger.debug(f"领取 {name}")
                else:
                    logger.debug(f"识别到但无法领取 {name}")
            else:
                logger.debug(f"未识别到 {name}")
        if claimed_count == 0:
            logger.debug("无可领取家具")
        else:
            logger.debug(f"领取完成，共 {claimed_count} 个")

        return CustomAction.RunResult(success=True)
