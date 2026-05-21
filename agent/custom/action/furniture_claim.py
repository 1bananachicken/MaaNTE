from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT

FURNITURE_LIST = [
    "仓鼠球",
    "棉棉",
    "破损的木箱",
]


@AgentServer.custom_action("furniture_claim")
class FurnitureClaim(CustomAction):
    def run(
        self, context: Context, _argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        for name in FURNITURE_LIST:
            image = controller.post_screencap().wait().get()
            result = context.run_recognition(
                "FurnitureOcrRec",
                image,
                pipeline_override={
                    "FurnitureOcrRec": {"recognition": {"param": {"expected": name}}}
                },
            )
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
                    PrintT(context, "furniture.claimed", name)
                else:
                    logger.debug(f"识别到但无法领取 {name}")
            else:
                logger.debug(f"未识别到 {name}")

        return CustomAction.RunResult(success=True)
