"""Check whether the current map position needs teleport."""

from __future__ import annotations

import importlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any

try:
    from maa.agent.agent_server import AgentServer
    from maa.context import Context
    from maa.custom_action import CustomAction
except ImportError:
    AgentServer = None
    Context = Any
    CustomAction = None


DEFAULT_NEAR_DISTANCE = 200.0
NEAR_MESSAGE = "距离过近，直接使用自动导航。"
FAR_MESSAGE = "距离较远，使用地图传送"


@dataclass(frozen=True)
class TargetPoint:
    id: str
    name: str
    point: tuple[int, int]
    threshold: float


@dataclass(frozen=True)
class TeleportDecision:
    current: tuple[int, int]
    target: tuple[int, int]
    distance: float
    need_teleport: bool

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


def point_xy(data: dict[str, Any]) -> tuple[int, int]:
    if "pixelX" in data and "pixelY" in data:
        return int(data["pixelX"]), int(data["pixelY"])
    if "x" in data and "y" in data:
        return int(data["x"]), int(data["y"])
    raise ValueError("point needs pixelX/pixelY or x/y")


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


def parse_target_point(params: dict[str, Any]) -> tuple[int, int]:
    table_point = load_target_point(params)
    if table_point is not None:
        return table_point.point

    target = params.get("target") or params.get("target_point")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return int(target[0]), int(target[1])
    if isinstance(target, dict) and "x" in target and "y" in target:
        return int(target["x"]), int(target["y"])
    return int(params["target_x"]), int(params["target_y"])


def parse_threshold(params: dict[str, Any]) -> float:
    if "threshold" in params:
        return float(params["threshold"])
    table_point = load_target_point(params)
    if table_point is not None:
        return table_point.threshold
    return DEFAULT_NEAR_DISTANCE


def check_teleport_required(
    frame: Any,
    target: tuple[int, int],
    *,
    threshold: float = DEFAULT_NEAR_DISTANCE,
    debug: bool = False,
) -> TeleportDecision | None:
    MapLocator = load_map_locator_class()
    logging.getLogger(MapLocator.__module__).setLevel(logging.WARNING)
    locator = MapLocator(debug=debug)
    try:
        result = locator.locate(frame)
        if not result.found or result.point is None:
            return None

        current_x, current_y = result.point
        target_x, target_y = target
        distance = math.hypot(current_x - target_x, current_y - target_y)
        return TeleportDecision(
            current=(current_x, current_y),
            target=(target_x, target_y),
            distance=distance,
            need_teleport=distance >= threshold,
        )
    finally:
        locator.close()


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
                debug = bool(params.get("debug", False))
            except Exception as exc:
                print(f"CheckTeleportRequired param invalid: {exc}")
                return CustomAction.RunResult(success=False)

            frame = context.tasker.controller.post_screencap().wait().get()
            if frame is None:
                print("read_failed")
                return CustomAction.RunResult(success=False)

            decision = check_teleport_required(
                frame,
                target,
                threshold=threshold,
                debug=debug,
            )
            if decision is None:
                print("not_found")
                return CustomAction.RunResult(success=False)

            try:
                from utils.maafocus import Print

                Print(context, decision.message)
            except Exception:
                print(decision.message)
            return CustomAction.RunResult(success=True)
