import json
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from .Common.utils import click_rect
from utils.logger import logger

_round_count = 0
_total_rounds = 3  # def


def _load_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        nested_value = value.get("custom_action_param")
        if nested_value is not None:
            return _load_params(nested_value)
        nested_action = value.get("action")
        if isinstance(nested_action, dict):
            nested_param = nested_action.get("param")
            if isinstance(nested_param, dict):
                nested_value = nested_param.get("custom_action_param")
                if nested_value is not None:
                    return _load_params(nested_value)
        return value
    if isinstance(value, list):
        for item in reversed(value):
            params = _load_params(item)
            if params:
                return params
        return {}
    if isinstance(value, (int, float)):
        return {"repeat_count": value}
    if isinstance(value, str) and value:
        try:
            params = json.loads(value)
        except json.JSONDecodeError:
            return {"repeat_count": value}
        if isinstance(params, (int, float)):
            return {"repeat_count": params}
        return params if isinstance(params, dict) else {}
    return {}


@AgentServer.custom_action("lucky_star_reset")
class LuckyStarReset(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _total_rounds
        _round_count = 0
        _total_rounds = 5
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("lucky_star_round_gate")
class LuckyStarRoundGate(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _round_count, _total_rounds

        params = _load_params(argv.custom_action_param)
        try:
            repeat_count = max(0, int(params.get("repeat_count", 5)))
        except (TypeError, ValueError):
            repeat_count = 5

        _round_count += 1
        total_rounds = repeat_count
        _total_rounds = total_rounds

        if _round_count > total_rounds:
            # logger.debug(
            #     "LuckyStar round limit reached: round=%d total=%d",
            #     _round_count,
            #     total_rounds,
            # )
            context.tasker.post_stop()
            return CustomAction.RunResult(success=True)
        elif _round_count == 1:
            next_nodes = ["PukaLandLuckyStarStart"]
        else:
            next_nodes = ["PukaLandLuckyStarCountdown"]

        logger.debug(
            "LuckyStar round gate: round=%d total=%d next=%s",
            _round_count,
            total_rounds,
            next_nodes,
        )
        context.override_next("PukaLandLuckyStarRound", next_nodes)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("lucky_star_retry_or_finish")
class LuckyStarRetryOrFinish(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        if _round_count >= _total_rounds:
            logger.debug(
                "LuckyStar finished: round=%d total=%d; stopping before retry",
                _round_count,
                _total_rounds,
            )
            context.tasker.post_stop()
            return CustomAction.RunResult(success=True)

        if argv.reco_detail is None:
            logger.warning("LuckyStar retry button recognition has no result box")
            return CustomAction.RunResult(success=False)

        click_rect(context.tasker.controller, argv.box, 0.005)
        return CustomAction.RunResult(success=True)
