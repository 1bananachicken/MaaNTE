"""Map teleport flow entrypoint."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from maa.agent.agent_server import AgentServer
    from maa.context import Context
    from maa.custom_action import CustomAction
    from maa.pipeline import JOCR, JRecognitionType
except ImportError:
    AgentServer = None
    Context = Any
    CustomAction = None
    JOCR = None
    JRecognitionType = None

from ..Common.logger import get_logger
from ..Common.utils import match_template_in_region
from .check_teleport_required import (
    DEFAULT_COORDINATE_TYPE,
    find_named_record,
    load_json_resource,
    point_xy,
    resource_base_path,
)

logger = get_logger(__name__)

DEFAULT_TELEPORT_POINTS_FILE = "map_teleport/teleport_points.json"
MAP_INDEX_ICON_TEMPLATE = "image/map_teleport/map_index_icon.png"
AREA_NEXT_BTN_TEMPLATE = "image/map_teleport/area_next_btn.png"
TELEPORT_ICON_TEMPLATE = "image/map_teleport/teleport_icon.png"

# 所有 ROI 都基于 1280x720；这些区域只覆盖地图索引和传送确认流程中需要看的小块。
MAP_INDEX_ICON_ROI = [1069, 626, 89, 74]
MAP_INDEX_TITLE_ROI = [907, 66, 121, 36]
AREA_NAME_ROI = [958, 127, 235, 40]
AREA_NEXT_BTN_ROI = [1203, 126, 44, 42]
RECOMMENDED_PLACE_ROI = [894, 175, 369, 468]
NEW_HERLAND_RECOMMENDED_PLACE_SWIPE_ROI = [897, 195, 351, 429]
TELEPORT_ICON_ROI = [894, 175, 369, 468]
TELEPORT_CONFIRM_POINT = [639, 361]
TELEPORT_BUTTON_ROI = [933, 620, 332, 45]

KEY_ESC = 27
KEY_M = 77

DEFAULT_TEMPLATE_THRESHOLD = 0.8
DEFAULT_OCR_THRESHOLD = 0.5
DEFAULT_MAX_AREA_SWITCHES = 15
DEFAULT_ACTION_DELAY = 0.5
DEFAULT_POST_MAP_OPEN_DELAY = 1.0
DEFAULT_FINAL_CONFIRM_DELAY = 1.0
DEFAULT_STILL_MAX_WAIT = 8.0
DEFAULT_STILL_INTERVAL = 0.25
DEFAULT_STILL_CONSECUTIVE = 2
DEFAULT_STILL_DIFF_THRESHOLD = 1.5
DEFAULT_SWIPE_DURATION_MS = 500


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


def _notify(context: Context | None, message: str) -> None:
    if context is None:
        print(message)
        return

    try:
        from utils.maafocus import Print

        Print(context, message)
    except Exception:
        logger.info("%s", message)


def _is_stopping(context: Context | None) -> bool:
    return bool(
        context is not None
        and getattr(getattr(context, "tasker", None), "stopping", False)
    )


def _template_path(relative_path: str) -> str:
    return str(resource_base_path() / relative_path)


def _load_template(relative_path: str) -> Any:
    template = cv2.imread(_template_path(relative_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError("template not found: %s" % relative_path)
    return template


def _normalize_frame(frame: Any) -> Any:
    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _screencap(context: Context) -> Any:
    controller = context.tasker.controller
    controller.post_screencap().wait()
    return _normalize_frame(controller.cached_image)


def _click_point(controller: Any, x: int, y: int, delay: float = 0.05) -> None:
    controller.post_touch_move(int(x), int(y)).wait()
    controller.post_touch_down(int(x), int(y)).wait()
    time.sleep(delay)
    controller.post_touch_up().wait()


def _click_rect(controller: Any, rect: list[int] | tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    _click_point(controller, int(x + w / 2), int(y + h / 2))


def _press_key_action(context: Context, key: int) -> bool:
    # 通过 Pipeline 的 ClickKey 发键，避免 CustomAction 内直接调用 controller 按键时的原生层异常。
    node_name = "__MapTeleportPressKey"
    try:
        result = context.run_action(
            node_name,
            pipeline_override={
                node_name: {
                    "action": {
                        "type": "ClickKey",
                        "param": {
                            "key": [
                                int(key),
                            ],
                        },
                    },
                    "pre_delay": 0,
                    "post_delay": 0,
                    "rate_limit": 0,
                }
            },
        )
    except Exception as exc:
        logger.error("MapTeleport press key failed: key=%s error=%s", key, exc)
        return False
    return result is not None


def _click_rect_action(context: Context, rect: list[int]) -> bool:
    # 关键确认点击同样交给 Pipeline Click，兼容 SeizeInput 等不同控制器实现。
    node_name = "__MapTeleportClick"
    try:
        result = context.run_action(
            node_name,
            pipeline_override={
                node_name: {
                    "action": {
                        "type": "Click",
                        "param": {
                            "target": rect,
                        },
                    },
                    "pre_delay": 0,
                    "post_delay": 0,
                    "rate_limit": 0,
                }
            },
        )
    except Exception as exc:
        logger.error("MapTeleport click failed: rect=%s error=%s", rect, exc)
        return False
    return result is not None


def _swipe_up_in_roi(context: Context, roi: list[int], *, duration: int) -> bool:
    x, y, w, h = roi
    start_x = int(x + w / 2)
    start_y = int(y + h * 0.82)
    end_x = start_x
    end_y = int(y + h * 0.18)
    try:
        context.tasker.controller.post_swipe(
            start_x,
            start_y,
            end_x,
            end_y,
            duration=duration,
        ).wait()
    except Exception as exc:
        logger.error("MapTeleport swipe failed: roi=%s error=%s", roi, exc)
        return False
    return True


def _box_to_rect(box: Any) -> list[int] | None:
    if box is None:
        return None
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    if all(hasattr(box, attr) for attr in ("x", "y", "w", "h")):
        return [int(box.x), int(box.y), int(box.w), int(box.h)]
    return None


def _detail_box(detail: Any) -> list[int] | None:
    if detail is None:
        return None

    rect = _box_to_rect(getattr(detail, "box", None))
    if rect is not None:
        return rect

    best_result = getattr(detail, "best_result", None)
    rect = _box_to_rect(getattr(best_result, "box", None))
    if rect is not None:
        return rect

    for item in getattr(detail, "filtered_results", None) or []:
        rect = _box_to_rect(getattr(item, "box", None))
        if rect is not None:
            return rect

    for item in getattr(detail, "all_results", None) or []:
        rect = _box_to_rect(getattr(item, "box", None))
        if rect is not None:
            return rect
    return None


def _wait_screen_still(
    context: Context,
    *,
    max_wait: float = DEFAULT_STILL_MAX_WAIT,
    interval: float = DEFAULT_STILL_INTERVAL,
    consecutive: int = DEFAULT_STILL_CONSECUTIVE,
    diff_threshold: float = DEFAULT_STILL_DIFF_THRESHOLD,
) -> Any | None:
    deadline = time.monotonic() + max(max_wait, interval)
    previous = _screencap(context)
    if previous is None:
        return None

    # 连续多帧变化很小时认为画面已稳定，避免动画期间误识别或误点。
    stable_count = 0
    last_frame = previous
    while time.monotonic() < deadline:
        if _is_stopping(context):
            return None

        time.sleep(max(interval, 0.05))
        current = _screencap(context)
        if current is None:
            continue
        last_frame = current

        prev_small = cv2.resize(previous, (160, 90), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(current, (160, 90), interpolation=cv2.INTER_AREA)
        diff = float(
            np.mean(
                cv2.absdiff(
                    cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY),
                )
            )
        )
        if diff <= diff_threshold:
            stable_count += 1
            if stable_count >= consecutive:
                return current
        else:
            stable_count = 0
        previous = current

    return last_frame


def _match_template(
    frame: Any,
    roi: list[int],
    template: Any,
    *,
    threshold: float = DEFAULT_TEMPLATE_THRESHOLD,
) -> tuple[bool, float, int, int]:
    return match_template_in_region(frame, roi, template, threshold)


def _find_template_matches(
    frame: Any,
    roi: list[int],
    template: Any,
    *,
    threshold: float = DEFAULT_TEMPLATE_THRESHOLD,
    max_results: int = 20,
) -> list[tuple[int, int, int, int, float]]:
    if frame is None or not isinstance(frame, np.ndarray):
        return []

    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return []

    region = frame[y1:y2, x1:x2]
    template_h, template_w = template.shape[:2]
    if region.shape[0] < template_h or region.shape[1] < template_w:
        return []

    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # 传送点图标会有多个，逐个取最高分并抑制同一图标周围的重复响应。
    matches: list[tuple[int, int, int, int, float]] = []
    suppress_x = max(1, template_w)
    suppress_y = max(1, template_h)
    while len(matches) < max_results:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break

        match_x = x1 + max_loc[0]
        match_y = y1 + max_loc[1]
        matches.append((match_x, match_y, template_w, template_h, float(max_val)))

        sx1 = max(0, max_loc[0] - suppress_x)
        sy1 = max(0, max_loc[1] - suppress_y)
        sx2 = min(result.shape[1], max_loc[0] + suppress_x)
        sy2 = min(result.shape[0], max_loc[1] + suppress_y)
        result[sy1:sy2, sx1:sx2] = -1.0

    matches.sort(key=lambda item: (item[1], item[0]))
    return matches


def _ocr_match(
    context: Context,
    frame: Any,
    text: str,
    roi: list[int],
    *,
    threshold: float = DEFAULT_OCR_THRESHOLD,
) -> Any | None:
    if JOCR is None or JRecognitionType is None:
        raise RuntimeError("maa.pipeline OCR is unavailable")
    if frame is None:
        return None

    detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(expected=[text], roi=tuple(roi), threshold=threshold),
        frame,
    )
    if detail is not None and getattr(detail, "hit", False):
        return detail
    return None


def _ensure_in_world(context: Context) -> bool:
    context.run_task("SceneAnyEnterWorld")
    frame = _wait_screen_still(context, max_wait=5.0)
    if frame is None:
        return False
    result = context.run_recognition("InWorld", frame)
    return bool(result and result.hit)


def _click_map_index(context: Context, template: Any, threshold: float) -> bool:
    frame = _wait_screen_still(context)
    matched, score, x, y = _match_template(
        frame,
        MAP_INDEX_ICON_ROI,
        template,
        threshold=threshold,
    )
    logger.debug("MapTeleport map index icon match=%s score=%.3f", matched, score)
    if not matched:
        return False

    _click_rect(
        context.tasker.controller,
        [x, y, template.shape[1], template.shape[0]],
    )
    return True


def _wait_ocr_text(
    context: Context,
    text: str,
    roi: list[int],
    *,
    threshold: float,
    max_wait: float = 5.0,
) -> Any | None:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if _is_stopping(context):
            return None
        frame = _wait_screen_still(context, max_wait=2.0)
        detail = _ocr_match(context, frame, text, roi, threshold=threshold)
        if detail is not None:
            return detail
        time.sleep(0.2)
    return None


def _switch_to_area(
    context: Context,
    area_name: str,
    area_next_template: Any,
    *,
    template_threshold: float,
    ocr_threshold: float,
    max_switches: int,
    action_delay: float,
) -> bool:
    for attempt in range(max_switches):
        if _is_stopping(context):
            return False

        # 地区名没命中时点击右上角切换按钮，最多翻 15 次避免卡死。
        frame = _wait_screen_still(context, max_wait=3.0)
        if _ocr_match(context, frame, area_name, AREA_NAME_ROI, threshold=ocr_threshold):
            logger.info("MapTeleport area matched: %s", area_name)
            return True

        matched, score, x, y = _match_template(
            frame,
            AREA_NEXT_BTN_ROI,
            area_next_template,
            threshold=template_threshold,
        )
        logger.debug(
            "MapTeleport area next attempt=%s match=%s score=%.3f",
            attempt + 1,
            matched,
            score,
        )
        if not matched:
            return False

        _click_rect(
            context.tasker.controller,
            [x, y, area_next_template.shape[1], area_next_template.shape[0]],
        )
        time.sleep(action_delay)

    return False


def _click_recommended_place(
    context: Context,
    *,
    ocr_threshold: float,
) -> bool:
    frame = _wait_screen_still(context, max_wait=3.0)
    detail = _ocr_match(
        context,
        frame,
        "推荐地点",
        RECOMMENDED_PLACE_ROI,
        threshold=ocr_threshold,
    )
    if detail is None:
        return False

    rect = _detail_box(detail) or RECOMMENDED_PLACE_ROI
    _click_rect(context.tasker.controller, rect)
    return True


def _adjust_recommended_place_list(
    context: Context,
    teleport_point: TeleportPoint,
    *,
    action_delay: float,
) -> bool:
    if teleport_point.area_name != "新赫兰德":
        return True

    # 新赫兰德的推荐地点列表需要先向上滚动，目标传送点才会进入后续图标匹配区域。
    if not _swipe_up_in_roi(
        context,
        NEW_HERLAND_RECOMMENDED_PLACE_SWIPE_ROI,
        duration=DEFAULT_SWIPE_DURATION_MS,
    ):
        return False
    time.sleep(action_delay)
    return True


def _click_teleport_icon(
    context: Context,
    teleport_point: TeleportPoint,
    teleport_icon_template: Any,
    *,
    template_threshold: float,
) -> bool:
    frame = _wait_screen_still(context, max_wait=5.0)
    matches = _find_template_matches(
        frame,
        TELEPORT_ICON_ROI,
        teleport_icon_template,
        threshold=template_threshold,
    )
    logger.info(
        "MapTeleport teleport icons matched: count=%s area_index=%s",
        len(matches),
        teleport_point.area_index,
    )
    if teleport_point.area_index < 1 or teleport_point.area_index > len(matches):
        return False

    x, y, w, h, _ = matches[teleport_point.area_index - 1]
    _click_rect(context.tasker.controller, [x, y, w, h])
    return True


def _click_teleport_button(
    context: Context,
    *,
    ocr_threshold: float,
) -> bool:
    frame = _wait_screen_still(context, max_wait=5.0)
    detail = _ocr_match(
        context,
        frame,
        "传送",
        TELEPORT_BUTTON_ROI,
        threshold=ocr_threshold,
    )
    if detail is None:
        return False

    return _click_rect_action(context, TELEPORT_BUTTON_ROI)


def _wait_teleport_loading(context: Context) -> bool:
    try:
        result = context.run_task("SceneLoadingType1")
    except Exception as exc:
        logger.error("MapTeleport wait loading failed: %s", exc)
        return False
    return result is not None


def run_map_teleport_flow(
    context: Context | None,
    teleport_point_id: str,
    *,
    points_file: str = DEFAULT_TELEPORT_POINTS_FILE,
    template_threshold: float = DEFAULT_TEMPLATE_THRESHOLD,
    ocr_threshold: float = DEFAULT_OCR_THRESHOLD,
    max_area_switches: int = DEFAULT_MAX_AREA_SWITCHES,
    action_delay: float = DEFAULT_ACTION_DELAY,
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
    _notify(context, message)

    if context is None:
        logger.error("MapTeleport needs Maa context to operate UI")
        return False

    controller = context.tasker.controller
    map_index_icon_template = _load_template(MAP_INDEX_ICON_TEMPLATE)
    area_next_template = _load_template(AREA_NEXT_BTN_TEMPLATE)
    teleport_icon_template = _load_template(TELEPORT_ICON_TEMPLATE)

    if not _ensure_in_world(context):
        _notify(context, "地图传送失败：当前未确认处于大世界界面")
        return False

    if not _press_key_action(context, KEY_M):
        _notify(context, "地图传送失败：打开地图按键失败")
        return False
    time.sleep(DEFAULT_POST_MAP_OPEN_DELAY)

    if not _click_map_index(context, map_index_icon_template, template_threshold):
        _notify(context, "地图传送失败：未找到地图索引按钮")
        return False

    if (
        _wait_ocr_text(
            context,
            "地图索引",
            MAP_INDEX_TITLE_ROI,
            threshold=ocr_threshold,
        )
        is None
    ):
        _notify(context, "地图传送失败：未进入地图索引")
        return False

    if not _switch_to_area(
        context,
        teleport_point.area_name,
        area_next_template,
        template_threshold=template_threshold,
        ocr_threshold=ocr_threshold,
        max_switches=max_area_switches,
        action_delay=action_delay,
    ):
        _notify(context, "地图传送失败：未找到地区 %s" % teleport_point.area_name)
        return False

    if not _click_recommended_place(context, ocr_threshold=ocr_threshold):
        _notify(context, "地图传送失败：未找到推荐地点")
        return False

    time.sleep(action_delay)
    if not _adjust_recommended_place_list(
        context,
        teleport_point,
        action_delay=action_delay,
    ):
        _notify(context, "地图传送失败：推荐地点列表滑动失败")
        return False

    if not _click_teleport_icon(
        context,
        teleport_point,
        teleport_icon_template,
        template_threshold=template_threshold,
    ):
        _notify(
            context,
            "地图传送失败：未找到第 %s 个传送图标" % teleport_point.area_index,
        )
        return False

    # 选中传送点后先退出地图指引，再点屏幕中央附近打开最终传送确认框。
    _wait_screen_still(context, max_wait=5.0)
    if not _press_key_action(context, KEY_ESC):
        _notify(context, "地图传送失败：退出地图按键失败")
        return False
    time.sleep(DEFAULT_FINAL_CONFIRM_DELAY)
    if not _click_rect_action(
        context,
        [TELEPORT_CONFIRM_POINT[0] - 1, TELEPORT_CONFIRM_POINT[1] - 1, 2, 2],
    ):
        _notify(context, "地图传送失败：确认传送点击失败")
        return False

    if not _click_teleport_button(context, ocr_threshold=ocr_threshold):
        _notify(context, "地图传送失败：未找到传送按钮")
        return False

    # 点击“传送”后交给通用加载节点等待，加载图消失即返回。
    time.sleep(action_delay)
    if not _wait_teleport_loading(context):
        _notify(context, "地图传送失败：等待传送加载结束失败")
        return False

    _notify(context, "地图传送流程已完成：%s" % teleport_point.name)
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
                template_threshold = float(
                    params.get("template_threshold", DEFAULT_TEMPLATE_THRESHOLD)
                )
                ocr_threshold = float(
                    params.get("ocr_threshold", DEFAULT_OCR_THRESHOLD)
                )
                max_area_switches = int(
                    params.get("max_area_switches", DEFAULT_MAX_AREA_SWITCHES)
                )
                action_delay = float(params.get("action_delay", DEFAULT_ACTION_DELAY))
            except Exception as exc:
                print("MapTeleportToPoint param invalid: %s" % exc)
                return CustomAction.RunResult(success=False)

            try:
                success = run_map_teleport_flow(
                    context,
                    teleport_point_id,
                    points_file=points_file,
                    template_threshold=template_threshold,
                    ocr_threshold=ocr_threshold,
                    max_area_switches=max_area_switches,
                    action_delay=action_delay,
                )
            except Exception as exc:
                print("MapTeleportToPoint failed: %s" % exc)
                return CustomAction.RunResult(success=False)
            return CustomAction.RunResult(success=success)
