import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction

if TYPE_CHECKING:
    from maa.context import Context

from ..Common.logger import get_logger
from .resource_paths import resource_base_path
from .route_model import (
    RouteSession,
    Waypoint,
    parse_route_segment_from_json_data,
    parse_waypoints_from_json_data,
)
from .waypoint_navigator import load_params

logger = get_logger(__name__)
DEFAULT_MAP_SIZE = (11264, 11264)
DEFAULT_TOLERANCE = 5.0
DEFAULT_FRAME_INTERVAL = 0.1
__all__ = [
    "LocalRouteNavigation",
    "LocalRouteNavigationAction",
    "LocalRouteNavigationUnitTestAction",
    "parse_route_waypoints",
    "resolve_route_json_path",
]


RouteJson = str | Path | dict[str, Any] | list[Any]


def resolve_route_json_path(json_path: str | Path) -> Path:
    path = Path(json_path).expanduser()
    if path.exists():
        return path

    routes_dir = resource_base_path().parent / "routes"
    file_name = path.name
    if not file_name:
        raise ValueError("route json path is empty")
    if not file_name.lower().endswith(".json"):
        file_name = f"{file_name}.json"

    candidate = routes_dir / file_name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"route json not found: {json_path}")


def parse_route_waypoints(
    route_json: Any,
    *,
    route_name: str = "",
    segment_index: int = 1,
    source_size: tuple[int, int] = DEFAULT_MAP_SIZE,
    target_size: tuple[int, int] = DEFAULT_MAP_SIZE,
) -> list[Waypoint]:
    if isinstance(route_json, dict):
        return parse_route_segment_from_json_data(
            route_json,
            route_name,
            segment_index,
            source_size,
            target_size,
        )
    return parse_waypoints_from_json_data(route_json, source_size, target_size)


class LocalRouteNavigation:
    """本地路线寻路器。

    一个实例持有一套 RouteRunner、定位器和方向推理器。需要在同一个函数中连续
    执行多个路线段时，先构造实例并加载整份路线 JSON，再多次调用 run_route()
    指定不同 segment，避免每段路线都重复初始化模型和定位资源。
    """

    def __init__(
        self,
        context: "Context",
        route_json: RouteJson | None = None,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        angle_backend: str = "auto",
        frame_interval: float = DEFAULT_FRAME_INTERVAL,
        debug: bool = False,
    ) -> None:
        from .route_runner import RouteRunner

        self.context = context
        self.route = RouteSession()
        self.runner = RouteRunner(
            context,
            self.route,
            angle_backend=angle_backend,
            tolerance=tolerance,
            frame_interval=frame_interval,
            debug=debug,
        )
        self._route_json: Any | None = None
        self._route_json_path: Path | None = None
        self._closed = False

        if route_json is not None:
            self.load_route_json(route_json)

    def load_route_json(self, route_json: RouteJson) -> None:
        if isinstance(route_json, (dict, list)):
            self._route_json = route_json
            self._route_json_path = None
            return

        route_json_path = resolve_route_json_path(route_json)
        with route_json_path.open("r", encoding="utf-8") as file:
            self._route_json = json.load(file)
        self._route_json_path = route_json_path

    def run_route(
        self,
        route_json: RouteJson | None = None,
        *,
        route_name: str = "",
        segment_index: int = 1,
    ) -> bool:
        if self._closed:
            raise RuntimeError("local route navigation is closed")

        self.runner.start()
        data, route_label = self._route_json_for_run(route_json)
        if data is None:
            raise ValueError("route json is not loaded")

        waypoints = parse_route_waypoints(
            data,
            route_name=route_name,
            segment_index=segment_index,
            source_size=DEFAULT_MAP_SIZE,
            target_size=self.runner.source_size(),
        )
        if not waypoints:
            logger.warning(
                "OnlineMapNavigation local route is empty: path=%s route=%s segment=%s",
                route_label,
                route_name,
                segment_index,
            )
            return False

        current_point = self.runner.update_current_frame()
        self.route.reset(waypoints, True, current_point)
        logger.info(
            "OnlineMapNavigation local route loaded: path=%s route=%s segment=%s "
            "waypoints=%s current=%s start_index=%s",
            route_label,
            route_name,
            segment_index,
            len(waypoints),
            current_point,
            self.route.payload()["currentIndex"],
        )
        status = self.runner.run_until_stopped(stop_when_route_done=True)
        return status == "arrived"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runner.close()

    def __enter__(self) -> "LocalRouteNavigation":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _route_json_for_run(
        self,
        route_json: RouteJson | None,
    ) -> tuple[Any | None, str | Path]:
        if route_json is None:
            return self._route_json, self._route_json_path or "<memory>"
        if isinstance(route_json, (dict, list)):
            return route_json, "<memory>"
        route_path = resolve_route_json_path(route_json)
        with route_path.open("r", encoding="utf-8") as file:
            return json.load(file), route_path


@AgentServer.custom_action("local_route_navigation")
class LocalRouteNavigationAction(CustomAction):
    def run(
        self, context: "Context", argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = load_params(argv.custom_action_param)
            json_path = params.get("json_path")
            if not json_path:
                raise ValueError("json_path is required")

            route_name = str(params.get("route_name", "")).strip()
            segment_index = int(params.get("segment_index", 1))
            tolerance = float(params.get("tolerance", DEFAULT_TOLERANCE))
            frame_interval = max(
                0.05,
                float(params.get("frame_interval", DEFAULT_FRAME_INTERVAL)),
            )
            angle_backend = str(params.get("angle_backend", "auto")).strip() or "auto"
            debug = bool(params.get("debug", False))
        except ValueError as exc:
            logger.error("LocalRouteNavigation param invalid: %s", exc)
            return CustomAction.RunResult(success=False)

        try:
            with LocalRouteNavigation(
                context,
                json_path,
                tolerance=tolerance,
                frame_interval=frame_interval,
                angle_backend=angle_backend,
                debug=debug,
            ) as navigator:
                success = navigator.run_route(
                    route_name=route_name,
                    segment_index=segment_index,
                )
            return CustomAction.RunResult(success=success)
        except Exception as exc:
            logger.error("LocalRouteNavigation failed: %s", exc)
            return CustomAction.RunResult(success=False)


@AgentServer.custom_action("local_route_navigation_unit_test")
class LocalRouteNavigationUnitTestAction(CustomAction):
    def run(
        self, context: "Context", argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = load_params(argv.custom_action_param)
            tolerance = float(params.get("tolerance", DEFAULT_TOLERANCE))
            frame_interval = max(
                0.05,
                float(params.get("frame_interval", DEFAULT_FRAME_INTERVAL)),
            )
            angle_backend = str(params.get("angle_backend", "auto")).strip() or "auto"
            debug = bool(params.get("debug", False))
        except ValueError as exc:
            logger.error("LocalRouteNavigationUnitTest param invalid: %s", exc)
            return CustomAction.RunResult(success=False)

        try:
            with LocalRouteNavigation(
                context,
                "penquan",
                tolerance=tolerance,
                frame_interval=frame_interval,
                angle_backend=angle_backend,
                debug=debug,
            ) as navigator:
                success = navigator.run_route(
                    route_name="penquan",
                    segment_index=1,
                )
            return CustomAction.RunResult(success=success)
        except Exception as exc:
            logger.error("LocalRouteNavigationUnitTest failed: %s", exc)
            return CustomAction.RunResult(success=False)
