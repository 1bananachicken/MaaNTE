import logging

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = logging.getLogger(__name__)

logger.warning(
    "auto_rhythm_repeat 已废弃，连打逻辑已迁移到 auto_rhythm_repeat_decision "
    "+ Pipeline JSON 编排。请勿再使用此 CustomAction。"
)


@AgentServer.custom_action("auto_rhythm_repeat")
class AutoRhythmRepeat(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.warning("auto_rhythm_repeat 已废弃，请使用 auto_rhythm_repeat_decision")
        return CustomAction.RunResult(success=True)