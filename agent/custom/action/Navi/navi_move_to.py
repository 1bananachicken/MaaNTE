from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils.maafocus import Print

from ..Common.logger import get_logger
from .path_navigator import PathNavigator, load_params, parse_bool

logger = get_logger(__name__)


@AgentServer.custom_action("navi_move_to")
class NaviMoveToAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = load_params(argv.custom_action_param)
            target = (int(params.get("target_x")), int(params.get("target_y")))
            navigator = PathNavigator(
                context,
                angle_backend=str(params.get("angle_backend", "auto")),
                tolerance=float(params.get("tolerance", 80.0)),
                max_duration=float(params.get("max_duration", 120.0)),
                debug=parse_bool(params.get("debug", False)),
            )
        except ValueError as exc:
            logger.warning("Navi move param invalid: %s", exc)
            Print(context, f"Navi move param invalid: {exc}")
            return CustomAction.RunResult(success=False)
        except Exception as exc:
            logger.error("Navi move init failed: %s", exc)
            Print(context, f"Navi move init failed: {exc}")
            return CustomAction.RunResult(success=False)

        try:
            logger.info("Navi move started: target=%s", target)
            return CustomAction.RunResult(success=navigator.move_to(target))
        except Exception as exc:
            logger.error("Navi move failed: %s", exc)
            Print(context, f"Navi move failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            navigator.close()
