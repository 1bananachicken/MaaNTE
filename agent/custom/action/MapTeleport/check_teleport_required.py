"""Check whether the current map position needs teleport."""

from __future__ import annotations

import importlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any, Callable

try:
    from maa.agent.agent_server import AgentServer
    from maa.context import Context
    from maa.custom_action import CustomAction
except ImportError:
    AgentServer = None
    Context = Any
    CustomAction = None


DEFAULT_NEAR_DISTANCE = 200.0
DEFAULT_POSITION_BACKEND = "auto"
DEFAULT_COORDINATE_TYPE = "world"
DEFAULT_COORDINATE_TIMEOUT = 1.5
DEFAULT_COORDINATE_INTERVAL = 0.1
NEAR_MESSAGE = "距离过近，直接使用自动导航。"
FAR_MESSAGE = "距离较远，使用地图传送"


@dataclass(frozen=True)
class TargetPoint:
    id: str
    name: str
    point: tuple[float, float]
    threshold: float


@dataclass(frozen=True)
class TeleportDecision:
    current: tuple[float, float]
    target: tuple[float, float]
    distance: float
    need_teleport: bool
    mode: str
    coordinate_type: str

    @property
    def message(self) -> str:
        return FAR_MESSAGE if self.need_teleport else NEAR_MESSAGE


def resource_base_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        assets_base = parent / "assets" / "resource" / "base"
        if assets_base.exists():
            return assets_base

        resource_base = parent / "resource" / "base"
        if resource_base.exists():
            return resource_base

    raise FileNotFoundError("Unable to locate resource/base directory")


def load_map_locator_class():
    if __package__:
        return importlib.import_module("..Navi.map_locator", __package__).MapLocator

    map_teleport_dir = Path(__file__).resolve().parent
    action_dir = map_teleport_dir.parent
    agent_dir = action_dir.parent.parent
    sys.path.insert(0, str(agent_dir))

    root_name = "_map_teleport_check_action"
    packages = {
        root_name: action_dir,
        f"{root_name}.Navi": action_dir / "Navi",
        f"{root_name}.Common": action_dir / "Common",
    }
    for package_name, package_path in packages.items():
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules.setdefault(package_name, package)

    return importlib.import_module(f"{root_name}.Navi.map_locator").MapLocator


def load_coordinate_position_provider_class():
    if __package__:
        return importlib.import_module(
            "..Navi.coordinate_position",
            __package__,
        ).CoordinatePositionProvider

    map_teleport_dir = Path(__file__).resolve().parent
    action_dir = map_teleport_dir.parent
    agent_dir = action_dir.parent.parent
    sys.path.insert(0, str(agent_dir))

    root_name = "_map_teleport_check_action"
    packages = {
        root_name: action_dir,
        f"{root_name}.Navi": action_dir / "Navi",
        f"{root_name}.Common": action_dir / "Common",
    }
    for package_name, package_path in packages.items():
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules.setdefault(package_name, package)

    return importlib.import_module(
        f"{root_name}.Navi.coordinate_position"
    ).CoordinatePositionProvider


def parse_params(custom_action_param: Any) -> dict[str, Any]:
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    return json.loads(custom_action_param)


def load_json_resource(relative_path: str | Path) -> Any:
    path = resource_base_path() / relative_path
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def point_xy(data: dict[str, Any]) -> tuple[float, float]:
    if "worldX" in data and "worldY" in data:
        return float(data["worldX"]), float(data["worldY"])
    if "rawX" in data and "rawY" in data:
        return float(data["rawX"]), float(data["rawY"])
    if "pixelX" in data and "pixelY" in data:
        return float(data["pixelX"]), float(data["pixelY"])
    if "x" in data and "y" in data:
        return float(data["x"]), float(data["y"])
    coordinate = data.get("coordinate")
    if isinstance(coordinate, dict) and "x" in coordinate and "y" in coordinate:
        return float(coordinate["x"]), float(coordinate["y"])
    raise ValueError("point needs worldX/worldY, pixelX/pixelY, or x/y")


def find_named_record(data: Any, collection_name: str, record_id: str) -> dict[str, Any]:
    records = data.get(collection_name) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"{collection_name} must be a list")

    normalized_id = str(record_id).strip()
    for record in records:
        if not isinstance(record, dict):
            continue
        keys = {
            str(record.get("id", "")).strip(),
            str(record.get("name", "")).strip(),
        }
        if normalized_id in keys:
            return record
    raise ValueError(f"{collection_name} record not found: {record_id}")


def load_target_point(params: dict[str, Any]) -> TargetPoint | None:
    point_id = str(params.get("point_id", "")).strip()
    if not point_id:
        return None

    table_path = str(
        params.get("points_file", "map_teleport/check_points.json")
    ).strip()
    data = load_json_resource(table_path)
    record = find_named_record(data, "points", point_id)
    threshold = float(record.get("threshold", DEFAULT_NEAR_DISTANCE))
    return TargetPoint(
        id=str(record.get("id", point_id)),
        name=str(record.get("name", record.get("id", point_id))),
        point=point_xy(record),
        threshold=threshold,
    )


def parse_target_point(params: dict[str, Any]) -> tuple[float, float]:
    table_point = load_target_point(params)
    if table_point is not None:
        return table_point.point

    target = params.get("target") or params.get("target_point")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return float(target[0]), float(target[1])
    if isinstance(target, dict) and "x" in target and "y" in target:
        return float(target["x"]), float(target["y"])
    return float(params["target_x"]), float(params["target_y"])


def parse_threshold(params: dict[str, Any]) -> float:
    if "threshold" in params:
        return float(params["threshold"])
    table_point = load_target_point(params)
    if table_point is not None:
        return table_point.threshold
    return DEFAULT_NEAR_DISTANCE


def parse_teleport_point_id(params: dict[str, Any]) -> str:
    return str(params.get("teleport_point_id", params.get("teleport_id", ""))).strip()


def visual_locate(
    frame: Any | None,
    *,
    get_frame: Callable[[], Any | None] | None = None,
    debug: bool = False,
) -> Any | None:
    if frame is None and get_frame is not None:
        frame = get_frame()
    if frame is None:
        return None

    MapLocator = load_map_locator_class()
    CoordinatePositionProvider = load_coordinate_position_provider_class()
    logging.getLogger(MapLocator.__module__).setLevel(logging.WARNING)
    locator = MapLocator(debug=debug)
    provider = CoordinatePositionProvider("map", debug=debug)
    try:
        return provider.locate(locator, frame)
    finally:
        try:
            provider.close()
        finally:
            locator.close()


def locate_current_position(
    *,
    frame: Any | None = None,
    get_frame: Callable[[], Any | None] | None = None,
    position_backend: str = DEFAULT_POSITION_BACKEND,
    coordinate_timeout: float = DEFAULT_COORDINATE_TIMEOUT,
    coordinate_interval: float = DEFAULT_COORDINATE_INTERVAL,
    debug: bool = False,
) -> Any | None:
    backend = str(position_backend or DEFAULT_POSITION_BACKEND).strip().lower()
    CoordinatePositionProvider = load_coordinate_position_provider_class()
    provider = CoordinatePositionProvider(backend, debug=debug)
    try:
        if provider.uses_visual_positioning():
            locator_result = visual_locate(frame, get_frame=get_frame, debug=debug)
            if locator_result is not None:
                return locator_result
            return None

        deadline = time.monotonic() + max(float(coordinate_timeout), 0.0)
        while True:
            result = provider.locate(None, None)
            if result.found and result.point is not None:
                return result
            if time.monotonic() >= deadline:
                break
            time.sleep(max(float(coordinate_interval), 0.05))

        if backend == "auto":
            return visual_locate(frame, get_frame=get_frame, debug=debug)
        return None
    finally:
        provider.close()


def check_teleport_required(
    frame_or_target: Any,
    target: tuple[float, float] | None = None,
    *,
    threshold: float = DEFAULT_NEAR_DISTANCE,
    position_backend: str = DEFAULT_POSITION_BACKEND,
    coordinate_type: str = DEFAULT_COORDINATE_TYPE,
    coordinate_timeout: float = DEFAULT_COORDINATE_TIMEOUT,
    coordinate_interval: float = DEFAULT_COORDINATE_INTERVAL,
    get_frame: Callable[[], Any | None] | None = None,
    debug: bool = False,
) -> TeleportDecision | None:
    frame = None
    if target is None:
        target = frame_or_target
    else:
        frame = frame_or_target

    result = locate_current_position(
        frame=frame,
        get_frame=get_frame,
        position_backend=position_backend,
        coordinate_timeout=coordinate_timeout,
        coordinate_interval=coordinate_interval,
        debug=debug,
    )
    if result is None or not result.found or result.point is None:
        return None

    current = location_point(result, coordinate_type)
    if current is None:
        return None

    current_x, current_y = current
    target_x, target_y = target
    distance = math.hypot(current_x - target_x, current_y - target_y)
    return TeleportDecision(
        current=current,
        target=(target_x, target_y),
        distance=distance,
        need_teleport=distance >= threshold,
        mode=str(result.mode),
        coordinate_type=normalize_coordinate_type(coordinate_type),
    )


def normalize_coordinate_type(coordinate_type: str) -> str:
    normalized = str(coordinate_type or DEFAULT_COORDINATE_TYPE).strip().lower()
    if normalized in {"world", "raw", "coordinate"}:
        return "world"
    if normalized in {"map", "pixel", "image"}:
        return "map"
    raise ValueError("coordinate_type must be world or map")


def location_point(result: Any, coordinate_type: str) -> tuple[float, float] | None:
    normalized = normalize_coordinate_type(coordinate_type)
    if normalized == "map":
        if result.point is None:
            return None
        return float(result.point[0]), float(result.point[1])

    raw = getattr(result, "raw_coordinate", None)
    if raw is None or len(raw) < 2:
        return None
    return float(raw[0]), float(raw[1])


if AgentServer is not None and CustomAction is not None:

    @AgentServer.custom_action("check_teleport_required")
    class CheckTeleportRequiredAction(CustomAction):
        def run(
            self, context: Context, argv: CustomAction.RunArg
        ) -> CustomAction.RunResult:
            try:
                params = parse_params(argv.custom_action_param)
                target = parse_target_point(params)
                threshold = parse_threshold(params)
                position_backend = (
                    str(params.get("position_backend", DEFAULT_POSITION_BACKEND)).strip()
                    or DEFAULT_POSITION_BACKEND
                )
                coordinate_type = (
                    str(params.get("coordinate_type", DEFAULT_COORDINATE_TYPE)).strip()
                    or DEFAULT_COORDINATE_TYPE
                )
                coordinate_timeout = float(
                    params.get("coordinate_timeout", DEFAULT_COORDINATE_TIMEOUT)
                )
                coordinate_interval = float(
                    params.get("coordinate_interval", DEFAULT_COORDINATE_INTERVAL)
                )
                teleport_point_id = parse_teleport_point_id(params)
                teleport_points_file = str(
                    params.get(
                        "teleport_points_file",
                        "map_teleport/teleport_points.json",
                    )
                ).strip()
                debug = bool(params.get("debug", False))
            except Exception as exc:
                print(f"CheckTeleportRequired param invalid: {exc}")
                return CustomAction.RunResult(success=False)

            def get_frame() -> Any | None:
                return context.tasker.controller.post_screencap().wait().get()

            try:
                decision = check_teleport_required(
                    target,
                    threshold=threshold,
                    position_backend=position_backend,
                    coordinate_type=coordinate_type,
                    coordinate_timeout=coordinate_timeout,
                    coordinate_interval=coordinate_interval,
                    get_frame=get_frame,
                    debug=debug,
                )
            except Exception as exc:
                print(f"CheckTeleportRequired failed: {exc}")
                return CustomAction.RunResult(success=False)

            if decision is None:
                print("not_found")
                return CustomAction.RunResult(success=False)

            try:
                from utils.maafocus import Print

                Print(context, decision.message)
            except Exception:
                print(decision.message)

            if decision.need_teleport:
                if not teleport_point_id:
                    print("CheckTeleportRequired failed: teleport_point_id is required")
                    return CustomAction.RunResult(success=False)
                try:
                    from .teleport_to_point import run_map_teleport_flow

                    success = run_map_teleport_flow(
                        context,
                        teleport_point_id,
                        points_file=teleport_points_file,
                    )
                except Exception as exc:
                    print(f"CheckTeleportRequired teleport failed: {exc}")
                    return CustomAction.RunResult(success=False)
                return CustomAction.RunResult(success=success)
            return CustomAction.RunResult(success=True)
