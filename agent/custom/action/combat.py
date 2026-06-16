import json
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

from ..combat import AutoCombat, CombatConfig


@AgentServer.custom_action("auto_combat")
class AutoCombatAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = _load_params(getattr(argv, "custom_action_param", None))
        config = CombatConfig.from_mapping(params)
        combat = AutoCombat(
            context.tasker.controller,
            should_stop=lambda: bool(context.tasker.stopping),
        )

        try:
            success = combat.run_loop(config)
        except Exception as exc:
            logger.exception("auto_combat failed: %s", exc)
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=success)


def _load_params(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse auto_combat custom_action_param: %r", raw)
            return {}
        return params if isinstance(params, dict) else {}
    return {}
