import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Waypoint = tuple[int, int]
SourceSize = tuple[int, int]
ONLINE_MAP_SIZE = (22528, 22528)
ONLINE_WORLD_ORIGIN_PIXEL = (11264.0, 11264.0)
ONLINE_PIXELS_PER_WORLD_UNIT = 44.0


@dataclass
class RouteSession:
    """自动寻路和 WebSocket 控制共享的可变路线状态。"""

    waypoints: list[Waypoint] = field(default_factory=list)
    active: bool = False
    current_index: int = 0
    status: str = "waiting"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "waypoints": [
                    {"pixelX": int(x), "pixelY": int(y)} for x, y in self.waypoints
                ],
                "active": self.active,
                "currentIndex": self.current_index,
                "status": self.status,
            }

    def reset(
        self,
        waypoints: list[Waypoint],
        start: bool,
        current_point: Waypoint | None,
    ) -> None:
        with self.lock:
            self.waypoints = waypoints
            self.active = bool(start and waypoints)
            self.current_index = (
                self.nearest_index(current_point)
                if self.active and current_point is not None
                else 0
            )
            self.status = "running" if self.active else "ready"

    def start(self, current_point: Waypoint | None) -> None:
        with self.lock:
            if not self.waypoints:
                self.active = False
                self.status = "empty"
                return
            self.current_index = (
                self.nearest_index(current_point)
                if current_point is not None
                else min(self.current_index, len(self.waypoints) - 1)
            )
            self.active = True
            self.status = "running"

    def advance(self) -> None:
        with self.lock:
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.active = False
                self.status = "arrived"

    def clear(self) -> None:
        with self.lock:
            self.waypoints.clear()
            self.current_index = 0
            self.active = False
            self.status = "cleared"

    def stop(self) -> None:
        with self.lock:
            self.active = False
            self.status = "stopped"

    def nearest_index(self, current_point: Waypoint) -> int:
        current_x, current_y = current_point
        return min(
            range(len(self.waypoints)),
            key=lambda index: (self.waypoints[index][0] - current_x) ** 2
            + (self.waypoints[index][1] - current_y) ** 2,
        )


def parse_waypoint(
    value: Any,
    source_size: SourceSize,
    target_size: SourceSize,
) -> Waypoint:
    if not isinstance(value, dict):
        raise ValueError("waypoint must be an object")
    if "pixelX" in value and "pixelY" in value:
        x = float(value["pixelX"])
        y = float(value["pixelY"])
    elif "target_x" in value and "target_y" in value:
        x = float(value["target_x"])
        y = float(value["target_y"])
    elif "lat" in value and "lng" in value:
        return parse_online_waypoint(value, target_size)
    elif "x" in value and "y" in value and value.get("coordinate") != "online":
        x = float(value["x"])
        y = float(value["y"])
    else:
        raise ValueError("waypoint needs pixelX/pixelY or x/y")

    source_size = parse_source_size(value, source_size)
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_w <= 0 or source_h <= 0:
        raise ValueError("waypoint source size must be positive")
    return int(round(x * target_w / source_w)), int(round(y * target_h / source_h))


def parse_online_waypoint(
    value: dict[str, Any],
    target_size: SourceSize,
) -> Waypoint:
    # maante-map stores route points as world coordinates named lat/lng.
    world_lat = float(value["lat"])
    world_lng = float(value["lng"])
    map_w, map_h = ONLINE_MAP_SIZE
    origin_x, origin_y = ONLINE_WORLD_ORIGIN_PIXEL
    target_w, target_h = target_size
    map_x = origin_x + world_lng * ONLINE_PIXELS_PER_WORLD_UNIT
    map_y = origin_y - world_lat * ONLINE_PIXELS_PER_WORLD_UNIT
    x = map_x * target_w / map_w
    y = map_y * target_h / map_h
    return int(round(x)), int(round(y))


def parse_source_size(value: dict[str, Any], default: SourceSize) -> SourceSize:
    if "sourceWidth" in value and "sourceHeight" in value:
        return int(value["sourceWidth"]), int(value["sourceHeight"])
    source_size = value.get("sourceSize")
    if isinstance(source_size, (list, tuple)) and len(source_size) >= 2:
        return int(source_size[0]), int(source_size[1])
    return default


def parse_waypoint_sequence(
    values: Any,
    source_size: SourceSize,
    target_size: SourceSize,
) -> list[Waypoint]:
    if not isinstance(values, list):
        raise ValueError("waypoints must be a list")
    return [parse_waypoint(value, source_size, target_size) for value in values]


def parse_waypoints_from_json_data(
    data: Any,
    source_size: SourceSize,
    target_size: SourceSize,
) -> list[Waypoint]:
    if isinstance(data, list):
        return parse_waypoint_sequence(data, source_size, target_size)
    if not isinstance(data, dict):
        raise ValueError("route json must be an object or list")

    route_data = data.get("route")
    if isinstance(route_data, dict) and isinstance(route_data.get("waypoints"), list):
        data = {**data, **route_data}

    if isinstance(data.get("waypoints"), list):
        values = data["waypoints"]
    elif isinstance(data.get("points"), list):
        values = data["points"]
    elif isinstance(data.get("path"), list):
        values = data["path"]
    else:
        raise ValueError("route json needs waypoints, points, or path")

    json_source_size = parse_source_size(data, source_size)
    return parse_waypoint_sequence(values, json_source_size, target_size)


def load_waypoints_from_json(
    path: str | Path,
    source_size: SourceSize,
    target_size: SourceSize,
) -> list[Waypoint]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        data = json.load(file)
    return parse_waypoints_from_json_data(data, source_size, target_size)


def parse_route_segment_from_json_data(
    data: Any,
    route_name: str,
    segment_index: int,
    source_size: SourceSize,
    target_size: SourceSize,
) -> list[Waypoint]:
    if not isinstance(data, dict):
        raise ValueError("route json must be an object")

    route_data = select_route(data, route_name)
    segments = route_data.get("segments")
    if not isinstance(segments, list):
        return parse_waypoints_from_json_data(route_data, source_size, target_size)
    if not segments:
        raise ValueError("route has no segments")

    index = normalize_segment_index(segment_index)
    if index >= len(segments):
        raise ValueError(
            f"segment index {segment_index} out of range, total={len(segments)}"
        )

    segment_data = segments[index]
    if not isinstance(segment_data, dict):
        raise ValueError("segment must be an object")

    segment_source_size = parse_source_size(
        segment_data,
        parse_source_size(route_data, parse_source_size(data, source_size)),
    )
    return parse_waypoints_from_json_data(
        segment_data,
        segment_source_size,
        target_size,
    )


def load_route_segment_from_json(
    path: str | Path,
    route_name: str,
    segment_index: int,
    source_size: SourceSize,
    target_size: SourceSize,
) -> list[Waypoint]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        data = json.load(file)
    return parse_route_segment_from_json_data(
        data,
        route_name,
        segment_index,
        source_size,
        target_size,
    )


def select_route(data: dict[str, Any], route_name: str) -> dict[str, Any]:
    routes = data.get("routes")
    if not isinstance(routes, list):
        return data
    if not routes:
        raise ValueError("route json has no routes")

    normalized_name = str(route_name).strip()
    if not normalized_name:
        route = routes[0]
        if isinstance(route, dict):
            return route
        raise ValueError("route must be an object")

    for route in routes:
        if not isinstance(route, dict):
            continue
        if normalized_name in {str(route.get("name", "")), str(route.get("id", ""))}:
            return route
    raise ValueError(f"route not found: {normalized_name}")


def normalize_segment_index(segment_index: int) -> int:
    index = int(segment_index)
    return 0 if index <= 1 else index - 1


def handle_route_message(
    message: dict[str, Any],
    route: RouteSession,
    source_size: SourceSize,
    current_point: Waypoint | None = None,
) -> dict[str, Any]:
    message_type = str(message.get("type", "")).strip()
    if message_type in ("navi-route-set", "route-set"):
        message_source_size = parse_source_size(message, source_size)
        route.reset(
            parse_waypoint_sequence(
                message.get("waypoints"),
                message_source_size,
                source_size,
            ),
            bool(message.get("start", False)),
            current_point,
        )
    elif message_type in ("navi-route-add", "route-add"):
        with route.lock:
            route.waypoints.append(
                parse_waypoint(
                    message,
                    parse_source_size(message, source_size),
                    source_size,
                )
            )
            if not route.active:
                route.status = "ready"
    elif message_type in ("navi-route-clear", "route-clear"):
        route.clear()
    elif message_type in ("navi-route-start", "route-start"):
        route.start(current_point)
    elif message_type in ("navi-route-stop", "route-stop"):
        route.stop()
    else:
        return {"type": "navi-route-ack", "ok": False, "message": "unknown type"}

    return {"type": "navi-route-ack", "ok": True, "route": route.payload()}
