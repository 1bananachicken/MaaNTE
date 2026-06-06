import json
import time

import cv2

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from ..Common.logger import get_logger
from .map_locator import MapLocationResult, MapLocator
from .angle_predictor import AnglePredictionResult, AnglePredictor
from .websocket_backend import NavigationWebSocketPublisher

logger = get_logger(__name__)


@AgentServer.custom_action("navi_websocket")
class NaviWebSocketAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = self.load_params(argv.custom_action_param)
        host = params.get("host", "0.0.0.0")
        port = params.get("port", 14514)
        debug = bool(params.get("debug", False))
        frame_interval = max(0.05, float(params.get("frame_interval", 0.1)))

        try:
            locator = MapLocator(
                debug=debug,
            )
            predictor = AnglePredictor(
                backend=params.get("angle_backend", "auto"),
                threshold=0.0,
                debug=debug,
            )
            angle_provider = predictor.provider_name()
            navigation_websocket = NavigationWebSocketPublisher(host, port)
            navigation_websocket.start()
        except Exception as exc:
            logger.error(f"Navi WebSocket init failed: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(
            f"Navi WebSocket action started: map={locator.big_map_path}, "
            f"angle_provider={angle_provider}, debug={debug}"
        )
        controller = context.tasker.controller
        last_location = MapLocationResult(False, None, None, 0.0, "init")
        last_angle = AnglePredictionResult(False, None, 0.0)

        try:
            while not context.tasker.stopping:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    continue

                last_location = locator.locate(frame)
                last_angle = predictor.predict(frame)
                navigation_websocket.publish_state(
                    last_location.point,
                    score=last_location.score,
                    mode=last_location.mode,
                    source_size=(locator.origin_w, locator.origin_h),
                    angle=last_angle.angle if last_angle.found else None,
                    angle_confidence=last_angle.confidence,
                )

                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                sleep_time = frame_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Navi WebSocket action failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            if debug:
                cv2.destroyAllWindows()

        return CustomAction.RunResult(success=last_location.found)

    @staticmethod
    def load_params(custom_action_param) -> dict:
        if not custom_action_param:
            return {}
        if isinstance(custom_action_param, dict):
            return custom_action_param
        try:
            params = json.loads(custom_action_param)
            return params if isinstance(params, dict) else {}
        except Exception as exc:
            logger.warning(f"Parse custom_action_param failed, use defaults: {exc}")
            return {}
