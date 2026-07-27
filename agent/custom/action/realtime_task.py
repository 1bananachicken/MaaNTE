import json
import time
from typing import List

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

from .realtime_navigation_state import consume_navigation_handoff


HOLDER_NODE_NAME = "__RealTimeTaskAction_Holder"
INITIAL_RETRY_DELAY = 0.05
MAX_RETRY_DELAY = 0.5


def _parse_nodes(custom_action_param: str) -> List[str]:
    if not custom_action_param:
        raise ValueError("RealTimeTaskAction: empty custom_action_param")

    params = json.loads(custom_action_param)
    if not isinstance(params, dict):
        raise ValueError(f"RealTimeTaskAction: invalid JSON object: {custom_action_param}")

    nodes = params.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError(f"RealTimeTaskAction: 'nodes' missing, not an array, or empty: {custom_action_param}")

    for v in nodes:
        if not isinstance(v, str):
            raise ValueError("RealTimeTaskAction: every entry in 'nodes' must be a string")

    return nodes


def _build_pipeline_override(nodes: List[str]) -> dict:
    return {HOLDER_NODE_NAME: {"next": nodes}}


def _get_check_interval(context: Context) -> float:
    node_data = context.get_node_data("RealTimeSleep") or {}
    try:
        interval_ms = max(0, int(node_data.get("post_delay", 200)))
    except (TypeError, ValueError):
        interval_ms = 200
    return interval_ms / 1000.0


def _wait_before_retry(context: Context, delay: float) -> None:
    """等待再次执行，同时保持对停止信号的响应。"""
    deadline = time.monotonic() + delay
    while not context.tasker.stopping:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(INITIAL_RETRY_DELAY, remaining))


@AgentServer.custom_action("RealTimeTaskAction")
class RealTimeTaskAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        nodes = _parse_nodes(argv.custom_action_param)
        pipeline_override = _build_pipeline_override(nodes)
        navigation_params = consume_navigation_handoff(argv.task_detail.task_id)
        if navigation_params is not None:
            return self.run_with_navigation(
                context,
                nodes,
                pipeline_override,
                navigation_params,
            )

        return self.run_realtime_loop(context, pipeline_override)

    @staticmethod
    def run_realtime_loop(
        context: Context,
        pipeline_override: dict,
    ) -> CustomAction.RunResult:
        retry_delay = INITIAL_RETRY_DELAY

        while not context.tasker.stopping:
            result = context.run_task(HOLDER_NODE_NAME, pipeline_override)
            if result is None:
                # 避免异常失败时无等待空转，持续占用 MaaFramework 任务线程。
                logger.debug(
                    "RealTimeTaskAction: RunTask returned None, retry in %.2fs",
                    retry_delay,
                )
                _wait_before_retry(context, retry_delay)
                retry_delay = min(MAX_RETRY_DELAY, retry_delay * 2)
                continue

            retry_delay = INITIAL_RETRY_DELAY

        logger.debug("RealTimeTaskAction: tasker stopping signal received, exit loop")
        return CustomAction.RunResult(success=True)

    @staticmethod
    def run_with_navigation(
        context: Context,
        nodes: List[str],
        pipeline_override: dict,
        navigation_params: dict,
    ) -> CustomAction.RunResult:
        from .Navi.online_map_navigation_action import OnlineMapNavigationAction

        check_interval = _get_check_interval(context)
        next_check_at = 0.0
        combined_override = dict(pipeline_override)
        combined_override["RealTimeSleep"] = {
            "pre_delay": 0,
            "post_delay": 0,
        }

        def run_realtime_once() -> None:
            nonlocal next_check_at
            now = time.monotonic()
            if now < next_check_at:
                return
            context.run_task(HOLDER_NODE_NAME, combined_override)
            next_check_at = time.monotonic() + check_interval

        logger.info(
            "RealTimeTaskAction: cooperative online navigation enabled, "
            "check_interval=%.3fs nodes=%s",
            check_interval,
            nodes,
        )
        navigation_result = OnlineMapNavigationAction.run_navigation(
            context,
            navigation_params,
            on_tick=run_realtime_once,
        )
        if not context.tasker.stopping and not navigation_result.success:
            logger.warning(
                "RealTimeTaskAction: online navigation exited unexpectedly; "
                "continuing realtime assistance"
            )
            return RealTimeTaskAction.run_realtime_loop(
                context,
                pipeline_override,
            )
        logger.debug("RealTimeTaskAction: combined task stopped")
        return CustomAction.RunResult(success=True)
