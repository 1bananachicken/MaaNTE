import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

logger = get_logger(__name__)
DEFAULT_MAP_SIZE = (11264, 11264)


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


def load_route_waypoints(
    json_path: str | Path,
    *,
    route_name: str = "",
    segment_index: int = 1,
    source_size: tuple[int, int] = DEFAULT_MAP_SIZE,
    target_size: tuple[int, int] = DEFAULT_MAP_SIZE,
) -> list[Waypoint]:
    route_path = resolve_route_json_path(json_path)
    with route_path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if isinstance(data, dict):
        return parse_route_segment_from_json_data(
            data,
            route_name,
            segment_index,
            source_size,
            target_size,
        )
    return parse_waypoints_from_json_data(data, source_size, target_size)


def run_route_from_json(
    context: "Context",
    json_path: str | Path,
    *,
    route_name: str = "",
    segment_index: int = 1,
    tolerance: float = 5.0,
    angle_backend: str = "auto",
    debug: bool = False,
) -> bool:
    from .route_runner import RouteRunner

    route = RouteSession()
    runner = RouteRunner(
        context,
        route,
        angle_backend=angle_backend,
        tolerance=tolerance,
        debug=debug,
    )

    try:
        runner.start()
        waypoints = load_route_waypoints(
            json_path,
            route_name=route_name,
            segment_index=segment_index,
            source_size=DEFAULT_MAP_SIZE,
            target_size=runner.source_size(),
        )
        if not waypoints:
            logger.warning("OnlineRoute local route is empty: path=%s", json_path)
            return False
        route.reset(waypoints, True, runner.current_point())
        logger.info(
            "OnlineRoute local route loaded: path=%s route=%s segment=%s waypoints=%s",
            resolve_route_json_path(json_path),
            route_name,
            segment_index,
            len(waypoints),
        )
        status = runner.run_until_stopped(stop_when_route_done=True)
        return status == "arrived"
    finally:
        runner.close()
