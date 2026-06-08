from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT

FURNITURE_RECOG_NODES = [
    ("FurnitureHamsterBall", "仓鼠球", "hamster_ball"),
    ("FurnitureFluff", "棉棉", "fluff"),
    ("FurnitureDamagedCrate", "破损的木箱", "damaged_crate"),
    ("FurnitureIntactCrate", "完整的木箱", "intact_crate"),
]


@AgentServer.custom_action("furniture_claim")
class FurnitureClaim(CustomAction):
    def run(
        self, context: Context, _argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        for node_name, name, msg_key in FURNITURE_RECOG_NODES:
            image = controller.post_screencap().wait().get()
            result = context.run_recognition(node_name, image)
            if result and result.box:
                roi = [result.box.x, result.box.y, result.box.w, result.box.h]
                result = context.run_recognition(
                    "FurnitureClaim",
                    image,
                    pipeline_override={
                        "FurnitureClaim": {"roi": roi}
                    },
                )
                if result.hit:
                    context.run_task(
                        "FurnitureClaim",
                        pipeline_override={
                            "FurnitureClaim": {"roi": roi}
                        },
                    )
                    PrintT(context, f"furniture.claimed.{msg_key}")
                else:
                    logger.debug(f"识别到但无法领取 {name}")
            else:
                logger.debug(f"未识别到 {name}")

        return CustomAction.RunResult(success=True)
