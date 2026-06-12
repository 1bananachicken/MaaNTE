from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from ..Common.logger import get_logger
from .waypoint_navigator import load_params, parse_bool
from .route_websocket_service import RouteWebSocketService
from .route_runner import RouteRunner
from .route_model import RouteSession

logger = get_logger(__name__)


@AgentServer.custom_action("online_route")
class OnlineRouteAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = load_params(argv.custom_action_param)
            params.update(self.load_option_params(context))
            host = str(params.get("host", "0.0.0.0"))
            port = int(params.get("port", 14514))
            tolerance = float(params.get("tolerance", 5.0))
            angle_backend = str(params.get("angle_backend", "auto"))
            debug = parse_bool(params.get("debug", False))
        except ValueError as exc:
            logger.error("OnlineRoute param invalid: %s", exc)
            return CustomAction.RunResult(success=False)

        route = RouteSession()
        runner = RouteRunner(
            context,
            route,
            angle_backend=angle_backend,
            tolerance=tolerance,
            debug=debug,
        )
        network = RouteWebSocketService(
            route,
            host=host,
            port=port,
            get_source_size=runner.source_size,
            get_current_point=runner.current_point,
        )
        runner.on_frame = network.publish_frame

        try:
            network.start()
            runner.start()
            logger.info("OnlineRoute service started: ws://%s:%s", host, port)
            runner.run_until_stopped(on_tick=network.publish_route)
            return CustomAction.RunResult(success=False)
        except Exception as exc:
            logger.error("OnlineRoute failed: %s", exc)
            return CustomAction.RunResult(success=False)
        finally:
            runner.close()
            network.stop()

    @staticmethod
    def load_option_params(context: Context) -> dict[str, Any]:
        params: dict[str, Any] = {}

        node_data = context.get_node_data("OnlineRouteAngleBackendConfig") or {}
        attach = node_data.get("attach")
        if isinstance(attach, dict) and attach.get("angle_backend") in {
            "auto",
            "directml",
            "cpu",
        }:
            params["angle_backend"] = attach["angle_backend"]

        node_data = context.get_node_data("OnlineRouteDebugConfig") or {}
        attach = node_data.get("attach")
        if isinstance(attach, dict) and "debug" in attach:
            params["debug"] = attach["debug"]

        return params
