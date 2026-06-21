"""Map teleport flow entrypoint.

The concrete UI operation will be filled in later. This module currently
resolves teleport point metadata and provides a stable Python flow boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from maa.agent.agent_server import AgentServer
    from maa.context import Context
    from maa.custom_action import CustomAction
except ImportError:
    AgentServer = None
    Context = Any
    CustomAction = None

from ..Common.logger import get_logger
from .check_teleport_required import (
    DEFAULT_COORDINATE_TYPE,
    find_named_record,
    load_json_resource,
    point_xy,
)

logger = get_logger(__name__)

DEFAULT_TELEPORT_POINTS_FILE = "map_teleport/teleport_points.json"


@dataclass(frozen=True)
class TeleportPoint:
    id: str
    name: str
    point: tuple[float, float]
    coordinate_type: str
    area_name: str
    area_index: int
    description: str


def parse_params(custom_action_param: Any) -> dict[str, Any]:
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    return json.loads(custom_action_param)


def load_teleport_point(
    teleport_point_id: str,
    *,
    points_file: str = DEFAULT_TELEPORT_POINTS_FILE,
) -> TeleportPoint:
    data = load_json_resource(points_file)
    record = find_named_record(data, "teleport_points", teleport_point_id)
    return TeleportPoint(
        id=str(record.get("id", teleport_point_id)),
        name=str(record.get("name", record.get("id", teleport_point_id))),
        point=point_xy(record),
        coordinate_type=str(record.get("coordinateType", DEFAULT_COORDINATE_TYPE)),
        area_name=str(record["areaName"]),
        area_index=int(record["areaIndex"]),
        description=str(record.get("description", "")),
    )


def run_map_teleport_flow(
    context: Context | None,
    teleport_point_id: str,
    *,
    points_file: str = DEFAULT_TELEPORT_POINTS_FILE,
) -> bool:
    teleport_point = load_teleport_point(
        teleport_point_id,
        points_file=points_file,
    )
    logger.info(
        "MapTeleport flow resolved: id=%s name=%s area=%s area_index=%s point=%s",
        teleport_point.id,
        teleport_point.name,
        teleport_point.area_name,
        teleport_point.area_index,
        teleport_point.point,
    )

    message = (
        "准备使用地图传送：%s（%s，第 %s 个）"
        % (teleport_point.name, teleport_point.area_name, teleport_point.area_index)
    )
    if context is not None:
        try:
            from utils.maafocus import Print

            Print(context, message)
        except Exception:
            print(message)
    else:
        print(message)

    # TODO: implement concrete map UI operation in the next step.
    return True


if AgentServer is not None and CustomAction is not None:

    @AgentServer.custom_action("map_teleport_to_point")
    class MapTeleportToPointAction(CustomAction):
        def run(
            self, context: Context, argv: CustomAction.RunArg
        ) -> CustomAction.RunResult:
            try:
                params = parse_params(argv.custom_action_param)
                teleport_point_id = str(
                    params.get("teleport_point_id", params.get("teleport_id", ""))
                ).strip()
                if not teleport_point_id:
                    raise ValueError("teleport_point_id is required")
                points_file = str(
                    params.get("teleport_points_file", DEFAULT_TELEPORT_POINTS_FILE)
                ).strip()
            except Exception as exc:
                print("MapTeleportToPoint param invalid: %s" % exc)
                return CustomAction.RunResult(success=False)

            try:
                success = run_map_teleport_flow(
                    context,
                    teleport_point_id,
                    points_file=points_file,
                )
            except Exception as exc:
                print("MapTeleportToPoint failed: %s" % exc)
                return CustomAction.RunResult(success=False)
            return CustomAction.RunResult(success=success)
