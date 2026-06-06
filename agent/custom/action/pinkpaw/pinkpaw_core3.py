from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from agent.custom.action.pinkpaw.pinkpaw_reward_logger import notify_pinkpaw_reward
except ImportError:
    from .pinkpaw_reward_logger import notify_pinkpaw_reward

VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
    "e": 0x45,
    "f": 0x46,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "esc": 0x1B,
    "lshift": 0xA0,
    "shift": 0x10,
}

MOUSE_VK = {
    "left": 0x01,
    "right": 0x02,
    "middle": 0x04,
}

REWARD_OCR_DELAY_MS = 3000
POST_REWARD_DELAY_MS = 7000
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_ROUTE_TIMING_SCALE = 1.0
DEFAULT_INTERACTION_PAUSE = 0.7
MIN_ROUTE_TIMING_SCALE = 0.25
MAX_ROUTE_TIMING_SCALE = 1.2
MIN_INTERACTION_PAUSE = 0.0
MAX_INTERACTION_PAUSE = 1.0
MAX_ROUTE_SLEEP_ADJUST = 0.25
ROUTE_SLEEP_ADJUST_RATIO_CAP = 0.08
ROUTE_SLEEP_BUSY_WAIT = 0.02
ROUTE_SLEEP_POLL_INTERVAL = 0.005
ROUTE_REWARD_CHECK_MIN_SLEEP = 0.5
WAIT_UNTIL_POLL_INTERVAL = 0.02
INTERAC_OCR_FALLBACK_INTERVAL = 1.0
FOCUS_LOG_NODE = "_PINKPAW_CORE3_FOCUS_"
DEFAULT_OCR_THRESHOLD = 0.2
TIMING_SENSITIVE_KEYS = {"w", "a", "s", "d", "lshift", "space", "e"}
TEAM_HEALTH_SLASH_ROI = [620, 654, 95, 42]
CURRENT_CHAR_MARKER_ROI = [1168, 164, 68, 36]
CURRENT_CHAR_MARKER_CORE_ROI = [1176, 172, 38, 16]
CURRENT_CHAR_SLOT_SPACING = 88
CURRENT_CHAR_MIN_SCORE = 16
CURRENT_CHAR_MIN_MARGIN = 5
CURRENT_CHAR_SLOT_WHITE_THRESHOLDS = [205, 188, 205, 205]
CURRENT_CHAR_SLOT_COLORED_THRESHOLDS = [170, 145, 170, 170]
CURRENT_CHAR_SLOT_SCORE_BONUS = [0, 4, 0, 0]
CURRENT_CHAR_SLOT_MIN_SCORE = [16, 12, 16, 16]
CURRENT_CHAR_SLOT_MIN_MARGIN = [5, 2, 5, 5]
CURRENT_CHAR_CORE_SCORE_WEIGHT = 3
CURRENT_CHAR_SLOT2_CORE_MIN_SCORE = 6
CURRENT_CHAR_SLOT2_CORE_MIN_MARGIN = 2
SWITCH_DEAD_SETTLE = 0.15
SWITCH_BLACK_SCREEN_EXTENSION = 0.5
SWITCH_CONFIRM_RETRY_COUNT = 1
SWITCH_CONFIRM_RETRY_WINDOW = 0.7
BLACK_SCREEN_MEAN_THRESHOLD = 18
BLACK_SCREEN_BRIGHT_PIXEL_THRESHOLD = 80
BLACK_SCREEN_BRIGHT_PIXEL_COUNT = 300
FAST_TEMPLATE_SAMPLE_LIMIT = 64
FAST_TEMPLATE_CANDIDATE_LIMIT = 5000
FAST_TEMPLATE_ANCHOR_TOLERANCE = 45
ENABLE_FAST_COLOR_RECO = True
FAST_TEMPLATE_RECO_NODES = {
    "PinkPawHeist_Core3_CheckInteractTemplateOnce",
    "PinkPawHeist_Core3_CheckSafeLockPromptOnce",
    "PinkPawHeist_Core3_CheckLockPickActiveTemplateOnce",
}

FAST_RECO_CONFIG = {
    "PinkPawHeist_Core3_CheckInteractPinkOnce": {
        "type": "color",
        "roi": [650, 240, 520, 460],
        "lower_bgr": [119, 71, 197],
        "upper_bgr": [133, 78, 221],
        "count": 80,
        "stride": 4,
    },
    "PinkPawHeist_Core3_CheckInteractTemplateOnce": {
        "type": "template",
        "roi": [680, 250, 430, 430],
        "templates": ["interactable.png", "heist_interac_lock_pick.png"],
        "threshold": 0.62,
        "cv_threshold": 0.88,
    },
    "PinkPawHeist_Core3_CheckSafeLockPromptOnce": {
        "type": "template",
        "roi": [680, 250, 430, 430],
        "templates": ["heist_interac_lock_pick.png"],
        "threshold": 0.56,
        "cv_threshold": 0.90,
    },
    "PinkPawHeist_Core3_CheckLockPickActiveTemplateOnce": {
        "type": "template",
        "roi": [720, 260, 360, 260],
        "templates": ["heist_lock_pick.png"],
        "threshold": 0.40,
        "cv_threshold": 0.86,
    },
}

_FAST_TEMPLATE_CACHE = {}
_FAST_IMAGE_DIR = None


class AbortException(Exception):
    pass


class EarlyExtractException(Exception):
    pass


class TaskerStoppedException(Exception):
    pass


@dataclass
class CharacterSwitchState:
    role: str
    keys: list[str]
    index: int = 0
    deadline: float = 0

    @property
    def current_key(self):
        return self.keys[self.index]

    def advance(self):
        self.index += 1
        return self.index < len(self.keys)


@dataclass
class OCRText:
    name: str
    text: str
    box: object = None
    confidence: float = 1.0
    score: float = 1.0


def _is_hit(result) -> bool:
    if result is None:
        return False
    status = getattr(result, "status", None)
    succeeded = getattr(status, "succeeded", None)
    if succeeded is not None:
        return bool(succeeded)
    if status is not None:
        return status == 0
    return bool(getattr(result, "hit", True))


def _norm_key(key: str) -> str:
    return str(key).lower()


def _parse_custom_action_param(argv: CustomAction.RunArg) -> dict:
    value = getattr(argv, "custom_action_param", None)
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception as exc:
        print(
            f"[PinkPawHeist/Core3] invalid custom_action_param: {value!r}, error: {exc}"
        )
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_timing_scale(value) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = DEFAULT_ROUTE_TIMING_SCALE
    return max(MIN_ROUTE_TIMING_SCALE, min(MAX_ROUTE_TIMING_SCALE, scale))


def _parse_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
    return bool(default)


def _parse_interaction_pause(value) -> float:
    try:
        pause = float(value)
    except (TypeError, ValueError):
        pause = DEFAULT_INTERACTION_PAUSE
    return max(MIN_INTERACTION_PAUSE, min(MAX_INTERACTION_PAUSE, pause))


def _get_fast_image_dir():
    global _FAST_IMAGE_DIR
    if _FAST_IMAGE_DIR is not None:
        return _FAST_IMAGE_DIR
    candidates = [
        Path.cwd() / "assets" / "resource" / "base" / "image" / "PinkPawHeist",
        Path(__file__).resolve().parents[4]
        / "assets"
        / "resource"
        / "base"
        / "image"
        / "PinkPawHeist",
    ]
    for candidate in candidates:
        if candidate.exists():
            _FAST_IMAGE_DIR = candidate
            return candidate
    _FAST_IMAGE_DIR = candidates[0]
    return _FAST_IMAGE_DIR


def _load_fast_template(name):
    if np is None or Image is None:
        return None
    if name in _FAST_TEMPLATE_CACHE:
        return _FAST_TEMPLATE_CACHE[name]
    path = _get_fast_image_dir() / name
    if not path.exists():
        _FAST_TEMPLATE_CACHE[name] = None
        return None
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    brightness = rgb.max(axis=2)
    saturation = brightness - rgb.min(axis=2)
    mask = (alpha >= 128) & ((brightness >= 80) | (saturation >= 40))
    if int(mask.sum()) == 0:
        mask = alpha >= 128
    coords = np.argwhere(mask)
    if coords.size == 0:
        _FAST_TEMPLATE_CACHE[name] = None
        return None
    if len(coords) > FAST_TEMPLATE_SAMPLE_LIMIT:
        scores = (
            brightness[coords[:, 0], coords[:, 1]].astype(np.int32)
            + saturation[coords[:, 0], coords[:, 1]].astype(np.int32) * 2
        )
        indices = np.argsort(scores)[-FAST_TEMPLATE_SAMPLE_LIMIT:]
        coords = coords[indices]
    gray = (
        rgb[:, :, 0].astype(np.float32) * 0.299
        + rgb[:, :, 1].astype(np.float32) * 0.587
        + rgb[:, :, 2].astype(np.float32) * 0.114
    )
    bgr = rgb[:, :, ::-1].astype(np.float32)
    cv_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    cv_mask = np.ascontiguousarray((alpha >= 128).astype(np.uint8) * 255)
    values = gray[coords[:, 0], coords[:, 1]].astype(np.float32)
    bgr_values = bgr[coords[:, 0], coords[:, 1]].astype(np.float32)
    template = {
        "name": name,
        "cv_bgr": cv_bgr,
        "cv_mask": cv_mask,
        "coords": coords.astype(np.int32),
        "bgr_values": bgr_values,
        "height": gray.shape[0],
        "width": gray.shape[1],
    }
    anchor_index = int(np.argmax(values))
    anchor_bgr = bgr_values[anchor_index]
    template["anchor_y"] = int(template["coords"][anchor_index, 0])
    template["anchor_x"] = int(template["coords"][anchor_index, 1])
    template["anchor_channel"] = int(np.argmax(anchor_bgr))
    template["anchor_value"] = int(anchor_bgr[template["anchor_channel"]])
    _FAST_TEMPLATE_CACHE[name] = template
    return template


def _as_bgr_image(image):
    if np is None or not isinstance(image, np.ndarray):
        return None
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    return image[:, :, :3]


def _crop_roi(image, roi):
    bgr = _as_bgr_image(image)
    if bgr is None:
        return None
    x, y, w, h = [int(v) for v in roi]
    ih, iw = bgr.shape[:2]
    x1 = max(0, min(iw, x))
    y1 = max(0, min(ih, y))
    x2 = max(x1, min(iw, x + w))
    y2 = max(y1, min(ih, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2]


def _scale_roi(roi, image):
    bgr = _as_bgr_image(image)
    if bgr is None:
        return roi
    ih, iw = bgr.shape[:2]
    sx = iw / DEFAULT_WIDTH
    sy = ih / DEFAULT_HEIGHT
    x, y, w, h = roi
    return [
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    ]


def _fast_color_match(image, cfg):
    roi = _crop_roi(image, cfg["roi"])
    if roi is None:
        return False
    stride = max(1, int(cfg.get("stride", 1)))
    if stride > 1:
        roi = roi[::stride, ::stride]
    lower = np.asarray(cfg["lower_bgr"], dtype=np.uint8)
    upper = np.asarray(cfg["upper_bgr"], dtype=np.uint8)
    mask = np.all((roi >= lower) & (roi <= upper), axis=2)
    count = max(1, int(cfg.get("count", 1)) // (stride * stride))
    return int(mask.sum()) >= count


def _fast_template_match(image, cfg):
    if cv2 is None:
        return None
    roi = _crop_roi(image, cfg["roi"])
    if roi is None:
        return None
    threshold = float(cfg.get("cv_threshold", cfg["threshold"]))
    roi = np.ascontiguousarray(roi)
    for name in cfg["templates"]:
        template = _load_fast_template(name)
        if template is None:
            continue
        templ = template["cv_bgr"]
        mask = template["cv_mask"]
        if roi.shape[0] < templ.shape[0] or roi.shape[1] < templ.shape[1]:
            continue
        scores = cv2.matchTemplate(roi, templ, cv2.TM_CCORR_NORMED, mask=mask)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            continue
        best = float(np.max(finite_scores))
        if best >= threshold:
            return True
    return None


def _fast_recognize_node(node_name, image):
    cfg = FAST_RECO_CONFIG.get(node_name)
    if cfg is None or np is None:
        return None
    if cfg["type"] == "color":
        if not ENABLE_FAST_COLOR_RECO:
            return None
        return _fast_color_match(image, cfg)
    if cfg["type"] == "template":
        if node_name not in FAST_TEMPLATE_RECO_NODES:
            return None
        if Image is None:
            return None
        return _fast_template_match(image, cfg)
    return None


def _value_to_pixel(value, total, default=0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return int(default)
    if 0 <= value <= 1:
        return int(round(value * total))
    return int(round(value))


def _calc_ocr_roi(image, x, y, to_x, to_y, width, height, box=None):
    if image is not None and hasattr(image, "shape") and len(image.shape) >= 2:
        frame_height, frame_width = image.shape[:2]
    else:
        frame_width, frame_height = DEFAULT_WIDTH, DEFAULT_HEIGHT

    if box is not None:
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            bx, by, bw, bh = box[:4]
            return [int(bx), int(by), max(1, int(bw)), max(1, int(bh))]
        bx = getattr(box, "x", None)
        by = getattr(box, "y", None)
        bw = getattr(box, "width", getattr(box, "w", None))
        bh = getattr(box, "height", getattr(box, "h", None))
        if None not in (bx, by, bw, bh):
            return [int(bx), int(by), max(1, int(bw)), max(1, int(bh))]

    roi_x = _value_to_pixel(x, frame_width)
    roi_y = _value_to_pixel(y, frame_height)
    if not width:
        width = float(to_x) - float(x)
    if not height:
        height = float(to_y) - float(y)
    roi_w = max(1, _value_to_pixel(width, frame_width, default=frame_width - roi_x))
    roi_h = max(1, _value_to_pixel(height, frame_height, default=frame_height - roi_y))

    roi_x = max(0, min(frame_width - 1, roi_x))
    roi_y = max(0, min(frame_height - 1, roi_y))
    roi_w = max(1, min(frame_width - roi_x, roi_w))
    roi_h = max(1, min(frame_height - roi_y, roi_h))
    return [roi_x, roi_y, roi_w, roi_h]


def _normalize_ocr_text(text: str) -> str:
    return str(text or "").strip()


def _ocr_text_matches(text: str, match) -> bool:
    text = _normalize_ocr_text(text)
    compact_text = re.sub(r"\s+", "", text)
    if match is None:
        return bool(text)
    matches = match if isinstance(match, list) else [match]
    for item in matches:
        if isinstance(item, str) and (item == text or item == compact_text):
            return True
        if isinstance(item, re.Pattern) and (
            re.search(item, text) or re.search(item, compact_text)
        ):
            return True
    return False


def _ocr_match_requests_text(match, text: str) -> bool:
    if match is None:
        return False
    matches = match if isinstance(match, list) else [match]
    for item in matches:
        if isinstance(item, str) and text in item:
            return True
        if isinstance(item, re.Pattern) and text in item.pattern:
            return True
    return False


def _ocr_result_items(result, match=None):
    items = []
    seen = set()
    for item in getattr(result, "all_results", None) or []:
        text = _normalize_ocr_text(getattr(item, "text", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        if not _ocr_text_matches(text, match):
            continue
        score = float(getattr(item, "score", 1.0) or 0.0)
        items.append(
            OCRText(
                name=text,
                text=text,
                box=getattr(item, "box", None),
                confidence=score,
                score=score,
            )
        )
    return items


def _normalize_ocr_frame(image):
    if image is None:
        return None
    if not hasattr(image, "shape") or len(image.shape) != 3:
        return image
    if image.shape[2] <= 3:
        return image
    return np.ascontiguousarray(image[:, :, :3]) if np is not None else image[:, :, :3]


class Core3ActionHelper:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mx, self.my = DEFAULT_WIDTH // 2, DEFAULT_HEIGHT // 2

    @property
    def controller(self):
        return getattr(getattr(self.ctx, "tasker", None), "controller", None)

    def is_stopping(self) -> bool:
        tasker = getattr(self.ctx, "tasker", None)
        if tasker is None:
            return False
        stopping = getattr(tasker, "stopping", False)
        if callable(stopping):
            stopping = stopping()
        return bool(stopping)

    def raise_if_stopped(self):
        if self.is_stopping():
            raise TaskerStoppedException(
                "PinkPawHeistScheme3Action stopped by Maa tasker."
            )

    def run_task(self, task_name, pipeline_override=None):
        self.raise_if_stopped()
        if pipeline_override is None:
            result = self.ctx.run_task(task_name)
        else:
            result = self.ctx.run_task(task_name, pipeline_override=pipeline_override)
        self.raise_if_stopped()
        return result

    def _call_key(self, node_type, key_str, extra=None):
        if node_type != "KeyUp":
            self.raise_if_stopped()
        vk = VK.get(_norm_key(key_str))
        if vk is None:
            return False
        controller = self.controller
        if controller is not None:
            if node_type == "KeyDown":
                controller.post_key_down(vk)
            elif node_type == "KeyUp":
                controller.post_key_up(vk)
            elif node_type == "ClickKey":
                if hasattr(controller, "post_click_key"):
                    controller.post_click_key(vk)
                else:
                    controller.post_key_down(vk)
                    time.sleep(0.02)
                    controller.post_key_up(vk)
            if node_type != "KeyUp":
                self.raise_if_stopped()
            return True
        param = {"key": vk}
        if extra:
            param.update(extra)
        node_name = f"PinkPawHeist_{node_type}"
        override = {node_name: {"action": {"type": node_type, "param": param}}}
        ret = self.ctx.run_task(node_name, pipeline_override=override) is not None
        if node_type != "KeyUp":
            self.raise_if_stopped()
        return ret

    def click_key(self, key_str):
        return self._call_key("ClickKey", key_str)

    def key_down(self, key_str):
        return self._call_key("KeyDown", key_str)

    def key_up(self, key_str):
        return self._call_key("KeyUp", key_str)

    def move_to(self, x, y, duration_ms=None):
        self.raise_if_stopped()
        x, y = int(x), int(y)
        dx, dy = x - self.mx, y - self.my
        if dx * dx + dy * dy < 4:
            self.mx, self.my = x, y
            return True
        if duration_ms is None:
            duration_ms = max(int((dx**2 + dy**2) ** 0.5 / 0.5), 50)
        override = {
            "PinkPawHeist_MouseMove": {
                "action": {
                    "type": "Swipe",
                    "param": {
                        "begin": [self.mx, self.my],
                        "end": [x, y],
                        "duration": duration_ms,
                        "only_hover": True,
                    },
                }
            }
        }
        ret = self.ctx.run_task("PinkPawHeist_MouseMove", pipeline_override=override)
        self.raise_if_stopped()
        if ret:
            self.mx, self.my = x, y
        return ret

    def click(self, x, y):
        self.raise_if_stopped()
        controller = self.controller
        if controller is not None and hasattr(controller, "post_click"):
            controller.post_click(int(x), int(y))
            self.raise_if_stopped()
            self.mx, self.my = int(x), int(y)
            return True
        self.move_to(x, y)
        override = {
            "PinkPawHeist_Click": {
                "action": {"type": "Click", "param": {"target": [int(x), int(y)]}}
            }
        }
        ret = (
            self.ctx.run_task("PinkPawHeist_Click", pipeline_override=override)
            is not None
        )
        self.raise_if_stopped()
        return ret

    def mouse_down(self, key="left"):
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_down(vk)

    def mouse_up(self, key="left"):
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_up(vk)

    def release_controls(self):
        for key in ("w", "a", "s", "d", "e", "f", "space", "lshift"):
            try:
                self.key_up(key)
            except Exception as exc:
                print(f"[PinkPawHeist/Core3] failed to release {key}: {exc}")
        controller = self.controller
        if controller is None:
            return
        for vk in MOUSE_VK.values():
            try:
                controller.post_key_up(vk).wait()
            except Exception as exc:
                print(f"[PinkPawHeist/Core3] failed to release mouse {vk}: {exc}")


class PinkPawHeistCore3Path:
    CONF_FIGHTER = "fighter"
    CONF_RUNNER = "runner"
    CONF_AVOIDER = "avoider"
    CONF_AVOID_MTH = "avoid_method"
    CONF_EARLY_EXTRACT_EXIT1 = "early_extract_exit1"
    CONF_EARLY_EXTRACT_EXIT2 = "early_extract_exit2"
    ROLE_FIGHTER = "fighter"
    ROLE_RUNNER = "runner"
    ROLE_AVOIDER = "avoider"
    AVOID_METHOD_DASH = "dash"
    AVOID_METHOD_ATTACK = "attack"
    SWITCH_CHECK_DURATION = 1.0
    QUICK_PICK_START_DELAY = 0.3
    QUICK_PICK_INTERVAL = 0.2

    def __init__(self, ctx: Context, params: dict | None = None):
        self.ctx = ctx
        self.ah = Core3ActionHelper(ctx)
        self.exit_state = {1: False, 2: False, 3: False, 4: False}
        self.avoid_methods = [self.AVOID_METHOD_DASH, self.AVOID_METHOD_ATTACK]
        avoid_method = (params or {}).get(self.CONF_AVOID_MTH, self.AVOID_METHOD_DASH)
        if avoid_method not in self.avoid_methods:
            self.log_warning(f"unknown avoid_method {avoid_method!r}, fallback to dash")
            avoid_method = self.AVOID_METHOD_DASH
        self.route_timing_scale = _parse_timing_scale(
            (params or {}).get("timing_scale", DEFAULT_ROUTE_TIMING_SCALE)
        )
        self.interaction_pause = _parse_interaction_pause(
            (params or {}).get("interaction_pause", DEFAULT_INTERACTION_PAUSE)
        )
        self.early_extract_exit = {
            1: _parse_bool((params or {}).get(self.CONF_EARLY_EXTRACT_EXIT1), False),
            2: _parse_bool((params or {}).get(self.CONF_EARLY_EXTRACT_EXIT2), False),
        }
        self.config = {
            self.CONF_FIGHTER: ["4", "1"],
            self.CONF_RUNNER: ["3"],
            self.CONF_AVOIDER: ["2"],
            self.CONF_AVOID_MTH: avoid_method,
        }
        self._dead_fighter_keys: list[str] = []
        self._current_fighter_key: str | None = None
        self._switch_state: CharacterSwitchState | None = None
        self._handling_switch_state = False
        self._next_switch_poll_at = 0.0
        self._held_keys: set[str] = set()
        self._quick_pick_active = False
        self._quick_pick_ready_at = 0.0
        self._next_quick_pick_at = 0.0
        self._last_action_at: dict[str, float] = {}
        self._interaction_watch_active = False
        self._interaction_watch_found = False
        self._checking_interaction = False
        self.last_check_reward_time = time.monotonic()
        self.check_reward_fail_count = 0
        self._round_label = "Core3"

    def log_info(self, *args):
        print("[PinkPawHeist/Core3]", *args)

    def log_warning(self, *args):
        print("[PinkPawHeist/Core3][WARN]", *args)

    def log_error(self, *args):
        print("[PinkPawHeist/Core3][ERROR]", *args)

    def log_round_info(self, message):
        self.log_info(f"{self._round_label}: {message}")
        self._log_to_frontend(str(message))

    def _log_to_frontend(self, message: str):
        try:
            self.ctx.run_action(
                FOCUS_LOG_NODE,
                pipeline_override={
                    FOCUS_LOG_NODE: {
                        "focus": {
                            "Node.Action.Starting": {
                                "content": f"[Core3] {message}",
                                "display": ["log", "toast"],
                            }
                        },
                        "action": "DoNothing",
                        "pre_delay": 0,
                        "post_delay": 0,
                    }
                },
            )
        except Exception:
            pass

    def _check_interval(self, name: str, interval: float) -> bool:
        if interval is None or interval < 0:
            return True
        now = time.monotonic()
        last = self._last_action_at.get(name, 0.0)
        if now - last < interval:
            return False
        self._last_action_at[name] = now
        return True

    def _poll_quick_pick(self):
        if not self._quick_pick_active:
            return
        now = time.monotonic()
        if now < self._quick_pick_ready_at or now < self._next_quick_pick_at:
            return
        self.ah.click_key("f")
        self._next_quick_pick_at = now + self.QUICK_PICK_INTERVAL

    def _has_timing_sensitive_key_held(self) -> bool:
        return bool(self._held_keys & TIMING_SENSITIVE_KEYS)

    def _check_still_in_heist(self):
        now = time.monotonic()
        if now - self.last_check_reward_time <= 2.0:
            return
        self.last_check_reward_time = now

        result = self.ah.run_task(
            "PinkPawHeist_CheckReward",
            pipeline_override={"PinkPawHeist_CheckReward": {"timeout": 100}},
        )
        if not _is_hit(result):
            self.check_reward_fail_count += 1
            self.log_warning(
                f"未检测到本局收益，连续失败 {self.check_reward_fail_count} 次"
            )
            if self.check_reward_fail_count >= 2:
                raise AbortException("PinkPawHeist_CheckReward 连续 2 次检测失败")
        else:
            self.check_reward_fail_count = 0

    def _scale_route_duration(self, duration: float) -> float:
        if duration <= 0 or self.route_timing_scale == 1.0:
            return max(duration, 0.0)

        wanted_adjust = duration * abs(1.0 - self.route_timing_scale)
        adaptive_cap = min(
            MAX_ROUTE_SLEEP_ADJUST,
            max(0.02, duration * ROUTE_SLEEP_ADJUST_RATIO_CAP),
        )
        adjust = min(wanted_adjust, adaptive_cap)
        if self.route_timing_scale < 1.0:
            return max(0.0, duration - adjust)
        return duration + adjust

    def sleep(self, timeout, check_reward=True, scaled=True):
        duration = max(float(timeout), 0.0)
        if scaled:
            duration = self._scale_route_duration(duration)
        target = time.perf_counter() + duration
        busy_from = target - ROUTE_SLEEP_BUSY_WAIT
        allow_reward_check = check_reward and duration >= ROUTE_REWARD_CHECK_MIN_SLEEP
        while time.perf_counter() < busy_from:
            self.ah.raise_if_stopped()
            self._poll_quick_pick()
            self._poll_character_switch()
            timing_sensitive = self._has_timing_sensitive_key_held()
            if allow_reward_check and not timing_sensitive:
                self._check_still_in_heist()
            if (
                self._interaction_watch_active
                and not self._interaction_watch_found
                and not self._checking_interaction
                and not timing_sensitive
            ):
                self._interaction_watch_found = self.find_interac()
            remaining = busy_from - time.perf_counter()
            time.sleep(max(0.0, min(ROUTE_SLEEP_POLL_INTERVAL, remaining)))
        while time.perf_counter() < target:
            if self.ah.is_stopping():
                self.ah.raise_if_stopped()
        self._poll_quick_pick()
        self._poll_character_switch()
        return True

    def next_frame(self):
        self.sleep(0.05)
        return True

    def send_key(
        self, key, down_time=0.02, interval=-1, after_sleep=0, action_name=None
    ):
        key = _norm_key(key)
        name = action_name or f"key:{key}"
        if not self._check_interval(name, interval):
            return False
        if key == "f":
            self.ah.click_key(key)
            if down_time and down_time > 0.06:
                self.sleep(down_time)
            if after_sleep:
                self.sleep(after_sleep)
            return True
        if down_time and down_time > 0.06:
            self.send_key_down(key)
            self.sleep(down_time)
            self.send_key_up(key)
        else:
            self.ah.click_key(key)
        if after_sleep:
            self.sleep(after_sleep)
        return True

    def send_key_down(self, key, after_sleep=0):
        key = _norm_key(key)
        if key == "f":
            if not self._quick_pick_active:
                self._quick_pick_ready_at = (
                    time.monotonic() + self.QUICK_PICK_START_DELAY
                )
                self._next_quick_pick_at = self._quick_pick_ready_at
            self._quick_pick_active = True
            return True
        self._held_keys.add(key)
        ret = self.ah.key_down(key)
        if after_sleep:
            self.sleep(after_sleep)
        return ret

    def send_key_up(self, key, after_sleep=0):
        key = _norm_key(key)
        if key == "f":
            self._quick_pick_active = False
            return True
        try:
            return self.ah.key_up(key)
        finally:
            self._held_keys.discard(key)
            if after_sleep:
                self.sleep(after_sleep)

    def sleep_send_key(self, time_out, key, interval=0.2):
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            self.send_key(key, interval=interval)
            self.sleep(0.01)

    def mouse_down(self, x=-1, y=-1, name=None, key="left"):
        self.ah.mouse_down(key=key)

    def mouse_up(self, name=None, key="left"):
        self.ah.mouse_up(key=key)

    def click(
        self,
        x=-1,
        y=-1,
        move_back=False,
        name=None,
        interval=-1,
        move=True,
        key="left",
        down_time=0.01,
        after_sleep=0,
    ):
        name = name or f"click:{key}"
        if not self._check_interval(name, interval):
            return False
        if x == -1:
            x = 0.5
        if y == -1:
            y = 0.5
        px = int(x * DEFAULT_WIDTH) if isinstance(x, float) and x <= 1 else int(x)
        py = int(y * DEFAULT_HEIGHT) if isinstance(y, float) and y <= 1 else int(y)
        if key == "left" and down_time <= 0.05:
            ret = self.ah.click(px, py)
        else:
            self.ah.move_to(px, py)
            self.ah.mouse_down(key=key)
            self.sleep(max(down_time, 0.01))
            self.ah.mouse_up(key=key)
            ret = True
        if after_sleep:
            self.sleep(after_sleep)
        return ret

    def wait_until(
        self,
        condition,
        time_out=0,
        pre_action=None,
        post_action=None,
        settle_time=-1,
        raise_if_not_found=False,
        **kwargs,
    ):
        timeout = 10.0 if not time_out or time_out <= 0 else float(time_out)
        deadline = time.monotonic() + timeout
        settled_at = None
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            if pre_action is not None:
                pre_action()
            found = bool(condition())
            if found:
                if post_action is not None:
                    post_action()
                if settle_time is not None and settle_time >= 0:
                    if settled_at is None:
                        settled_at = time.monotonic()
                    if time.monotonic() - settled_at >= settle_time:
                        return True
                else:
                    return True
            else:
                settled_at = None
            self.sleep(WAIT_UNTIL_POLL_INTERVAL, check_reward=False, scaled=False)
        if raise_if_not_found:
            raise AbortException("timeout for wait_until")
        return False

    def wait_team_ui_settle(self):
        self.wait_until(
            lambda: not self.is_in_team(),
            time_out=1,
            raise_if_not_found=False,
        )
        self.wait_until(
            self.is_in_team,
            time_out=30,
            settle_time=0.25,
            raise_if_not_found=False,
        )
        self.sleep(0.1, check_reward=False)
        return True

    def _is_black_screen_in_image(self, image):
        bgr = _as_bgr_image(image)
        if bgr is None:
            return False
        sample = bgr[::8, ::8]
        if sample.size == 0:
            return False
        max_ch = sample.max(axis=2)
        return (
            float(max_ch.mean()) <= BLACK_SCREEN_MEAN_THRESHOLD
            and int((max_ch >= BLACK_SCREEN_BRIGHT_PIXEL_THRESHOLD).sum())
            <= BLACK_SCREEN_BRIGHT_PIXEL_COUNT
        )

    def _is_in_team_in_image(self, image):
        if np is None:
            return True
        roi = _crop_roi(image, _scale_roi(TEAM_HEALTH_SLASH_ROI, image))
        if roi is None:
            return False
        max_ch = roi.max(axis=2)
        min_ch = roi.min(axis=2)
        bright = (max_ch >= 175) & ((max_ch - min_ch) <= 95)
        return int(bright.sum()) >= 10

    def is_in_team(self):
        image = self._screencap()
        if image is None:
            return True
        return self._is_in_team_in_image(image)

    def _current_char_roi_score(self, image, roi, index):
        crop = _crop_roi(image, _scale_roi(roi, image))
        if crop is None:
            return 0
        max_ch = crop.max(axis=2)
        min_ch = crop.min(axis=2)
        sat = max_ch - min_ch
        white_threshold = CURRENT_CHAR_SLOT_WHITE_THRESHOLDS[index]
        colored_threshold = CURRENT_CHAR_SLOT_COLORED_THRESHOLDS[index]
        white = (max_ch >= white_threshold) & (sat <= 65)
        colored = (max_ch >= colored_threshold) & (sat >= 55)
        return int((white | colored).sum())

    def _current_char_scores(self, image):
        if np is None:
            return [0, 0, 0, 0]
        scores = []
        for index in range(4):
            broad_roi = list(CURRENT_CHAR_MARKER_ROI)
            broad_roi[1] += CURRENT_CHAR_SLOT_SPACING * index
            score = self._current_char_roi_score(image, broad_roi, index)
            score += CURRENT_CHAR_SLOT_SCORE_BONUS[index]
            scores.append(score)
        return scores

    def _current_char_core_scores(self, image):
        if np is None:
            return [0, 0, 0, 0]
        scores = []
        for index in range(4):
            core_roi = list(CURRENT_CHAR_MARKER_CORE_ROI)
            core_roi[1] += CURRENT_CHAR_SLOT_SPACING * index
            scores.append(
                self._current_char_roi_score(image, core_roi, index)
                * CURRENT_CHAR_CORE_SCORE_WEIGHT
            )
        return scores

    def _is_current_char_score_accepted(self, scores, index):
        if not scores or not 0 <= index < len(scores):
            return False
        target_score = scores[index]
        other_scores = [score for idx, score in enumerate(scores) if idx != index]
        best_other = max(other_scores) if other_scores else 0
        min_score = CURRENT_CHAR_SLOT_MIN_SCORE[index]
        min_margin = CURRENT_CHAR_SLOT_MIN_MARGIN[index]
        return target_score >= min_score and target_score - best_other >= min_margin

    def _is_slot2_core_score_accepted(self, image):
        scores = self._current_char_core_scores(image)
        target_score = scores[1]
        best_other = max(score for idx, score in enumerate(scores) if idx != 1)
        return (
            target_score >= CURRENT_CHAR_SLOT2_CORE_MIN_SCORE
            and target_score - best_other >= CURRENT_CHAR_SLOT2_CORE_MIN_MARGIN
        )

    def get_current_char_index(self, image=None):
        if image is None:
            image = self._screencap()
        if image is None:
            return -1
        scores = self._current_char_scores(image)
        if not scores:
            return -1
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if self._is_current_char_score_accepted(scores, best_idx):
            return best_idx
        return -1

    def is_char_at_index(self, index, image=None):
        if image is None:
            image = self._screencap()
        if image is None:
            return False
        index = int(index)
        if self._is_current_char_score_accepted(
            self._current_char_scores(image), index
        ):
            return True
        if index == 1:
            return self._is_slot2_core_score_accepted(image)
        return False

    def ensure_in_team(self, time_out=2.0):
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            if self.is_in_team():
                return True
            self.send_key("esc", action_name="ensure_in_team", interval=0.3)
            self.sleep(0.05, check_reward=False, scaled=False)
        return self.is_in_team()

    def _run_check_node(self, node_name, timeout=1.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            if self._recognize_once(node_name):
                return True
            time.sleep(0.01)
        return False

    def _screencap(self):
        controller = getattr(getattr(self.ctx, "tasker", None), "controller", None)
        if controller is None:
            return None
        return controller.post_screencap().wait().get()

    def _recognize_once(self, node_name, image=None):
        self.ah.raise_if_stopped()
        if image is None:
            image = self._screencap()
        if image is None:
            return _is_hit(self.ah.run_task(node_name))
        fast_result = _fast_recognize_node(node_name, image)
        if fast_result is not None:
            return fast_result
        result = self.ctx.run_recognition(node_name, image)
        self.ah.raise_if_stopped()
        return _is_hit(result)

    def _find_interac_in_image(self, image, include_ocr=False):
        if self._recognize_once("PinkPawHeist_Core3_CheckInteractPinkOnce", image):
            return True
        if self._recognize_once("PinkPawHeist_Core3_CheckInteractTemplateOnce", image):
            return True
        if not include_ocr:
            return False
        return any(
            self._recognize_once(node, image)
            for node in (
                "PinkPawHeist_Core3_CheckInteractOnce",
                "PinkPawHeist_CheckDoorOnce",
                "PinkPawHeist_CheckGateOnce",
                "PinkPawHeist_CheckGate2Once",
                "PinkPawHeist_CheckEvacuateOnce",
            )
        )

    def find_interac(self, include_ocr=False):
        self._checking_interaction = True
        try:
            image = self._screencap()
            return self._find_interac_in_image(image, include_ocr=include_ocr)
        finally:
            self._checking_interaction = False

    def ocr_once(
        self,
        x=0,
        y=0,
        to_x=1,
        to_y=1,
        width=0,
        height=0,
        name=None,
        box=None,
        match=None,
        threshold=0,
        frame=None,
        target_height=0,
        time_out=0,
        post_action=None,
        raise_if_not_found=False,
        log=False,
        screenshot=False,
        lib="default",
    ):
        image = frame if frame is not None else self._screencap()
        image = _normalize_ocr_frame(image)
        if image is None:
            return None
        roi = _calc_ocr_roi(image, x, y, to_x, to_y, width, height, box=box)
        ocr_threshold = DEFAULT_OCR_THRESHOLD if not threshold else float(threshold)
        try:
            result = self.ctx.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(roi=roi, threshold=ocr_threshold),
                image,
            )
        except Exception as exc:
            self.log_warning(f"OCR failed roi={roi} match={match}: {exc}")
            return None

        items = _ocr_result_items(result, match=match)
        if log:
            all_texts = [
                _normalize_ocr_text(getattr(item, "text", ""))
                for item in getattr(result, "all_results", None) or []
            ]
            self.log_info(f"OCR roi={roi} match={match} hit={items} all={all_texts}")
        return items or None

    def wait_ocr(
        self,
        x=0,
        y=0,
        to_x=1,
        to_y=1,
        width=0,
        height=0,
        name=None,
        box=None,
        match=None,
        threshold=0,
        frame=None,
        target_height=0,
        time_out=0,
        post_action=None,
        raise_if_not_found=False,
        log=False,
        screenshot=False,
        settle_time=-1,
        lib="default",
    ):
        timeout = 1.5 if not time_out or time_out <= 0 else float(time_out)
        deadline = time.monotonic() + timeout
        settled_at = None
        last_items = None
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            items = self.ocr_once(
                x=x,
                y=y,
                to_x=to_x,
                to_y=to_y,
                width=width,
                height=height,
                name=name,
                box=box,
                match=match,
                threshold=threshold,
                frame=frame,
                target_height=target_height,
                log=log,
                screenshot=screenshot,
                lib=lib,
            )
            if items:
                last_items = items
                if settle_time is not None and settle_time >= 0:
                    if settled_at is None:
                        settled_at = time.monotonic()
                    if time.monotonic() - settled_at >= settle_time:
                        if post_action:
                            post_action()
                        return last_items
                else:
                    if post_action:
                        post_action()
                    return items
            else:
                settled_at = None
            if frame is not None:
                break
            self.sleep(WAIT_UNTIL_POLL_INTERVAL, check_reward=False, scaled=False)
        if frame is None and _ocr_match_requests_text(match, "开门"):
            if self._run_check_node("PinkPawHeist_CheckDoorOnce", timeout=0.25):
                items = [OCRText(name="开门", text="开门")]
                if post_action:
                    post_action()
                return items
        if raise_if_not_found:
            raise AbortException("timeout for wait_ocr")
        return None

    def start_interaction_watch(self):
        self._interaction_watch_active = True
        self._interaction_watch_found = False
        return True

    def stop_interaction_watch(self):
        self._interaction_watch_active = False
        self._interaction_watch_found = False
        return True

    def is_lock_pick_active_fast(self):
        image = self._screencap()
        return self._is_lock_pick_active_fast_in_image(image)

    def _is_lock_pick_active_fast_in_image(self, image):
        return self._recognize_once("PinkPawHeist_Core3_CheckLockPickActiveOnce", image)

    def is_lock_pick_active(self):
        return self.is_lock_pick_active_fast()

    def wait_lock_pick_active(self, time_out=2, settle_time=-1):
        if self.wait_until(
            self.is_lock_pick_active_fast,
            time_out=time_out,
            settle_time=settle_time,
        ):
            return True
        return self.is_lock_pick_active()

    def is_safe_lock_pick_active(self):
        image = self._screencap()
        if self._is_lock_pick_active_fast_in_image(image):
            return True
        return self._recognize_once(
            "PinkPawHeist_Core3_CheckLockPickActiveTemplateOnce", image
        )

    def wait_safe_lock_pick_active(self, time_out=2, settle_time=-1):
        return self.wait_until(
            self.is_safe_lock_pick_active,
            time_out=time_out,
            settle_time=settle_time,
        )

    def has_safe_lock_prompt(self):
        image = self._screencap()
        return self._recognize_once("PinkPawHeist_Core3_CheckSafeLockPromptOnce", image)

    def wait_for_interac(self, time_out=10, include_ocr_fallback=True):
        if self.wait_until(self.find_interac, time_out=time_out):
            return True
        if include_ocr_fallback:
            return self.find_interac(include_ocr=True)
        return False

    def wait_and_interact(
        self,
        direction=None,
        interact=True,
        key_up_sleep=None,
        is_lock=False,
        time_out=10,
    ):
        timeout = 10.0 if not time_out or time_out <= 0 else float(time_out)
        lock_min_done_at = 0.0

        def start_lock_min_timer():
            nonlocal lock_min_done_at
            if is_lock and lock_min_done_at <= 0:
                lock_min_done_at = time.monotonic() + timeout

        def remaining_min_time():
            if not is_lock or lock_min_done_at <= 0:
                return 0.0
            return max(0.0, lock_min_done_at - time.monotonic())

        def wait_until_min_time():
            remaining = remaining_min_time()
            if remaining > 0:
                self.sleep(remaining, check_reward=False, scaled=False)

        def press_interact():
            start_lock_min_timer()
            self.send_key("f", interval=0.5)

        ret = self.wait_for_interac(time_out=timeout)
        if interact and direction is not None:
            self.send_key_up(direction)
            if key_up_sleep is None:
                key_up_sleep = self.interaction_pause
            self.sleep(key_up_sleep, check_reward=False, scaled=False)
        if not ret:
            raise AbortException("timeout for wait_and_interact")
        if not interact:
            wait_until_min_time()
            return True
        interaction_closed = self.wait_until(
            lambda: not self.find_interac(),
            pre_action=press_interact,
            time_out=max(2.0, timeout if is_lock else 0.001),
        )
        if is_lock:
            lock_started = self.wait_until(
                self.is_lock_pick_active_fast,
                time_out=max(2.0, remaining_min_time(), 0.001),
            )
            if lock_started:
                lock_finished = self.wait_until(
                    lambda: not self.is_lock_pick_active_fast(),
                    time_out=max(10.0, remaining_min_time(), 0.001),
                    settle_time=0.15,
                )
                if not lock_finished:
                    wait_until_min_time()
                    self.log_warning("未确认撬锁结束，按保底时间等待后继续")
            else:
                self.log_warning("未确认撬锁开始，按保底时间等待后继续确认")
            wait_until_min_time()
            if not interaction_closed and self.find_interac(include_ocr=True):
                self.log_warning("锁交互提示仍可见，继续路线")
            return True
        wait_until_min_time()
        return True

    def loot_safes_while_walking(
        self, direction=None, min_walk_time=0, time_out=10, hold=False, send_pick=False
    ):
        start_time = time.monotonic()
        deadline = start_time + time_out
        earliest_lock_pick_time = start_time + min_walk_time
        if direction is not None:
            self.send_key_down(direction)
        pick_started = False

        def wait_until_pick_time():
            nonlocal pick_started
            remaining = earliest_lock_pick_time - time.monotonic()
            if remaining > 0:
                self.sleep(remaining)
            if send_pick and not pick_started:
                self.send_key_down("f")
                pick_started = True

        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                if send_pick and not pick_started and now >= earliest_lock_pick_time:
                    self.send_key_down("f")
                    pick_started = True
                if self.has_safe_lock_prompt():
                    if now < earliest_lock_pick_time:
                        wait_until_pick_time()
                    lock_pick_start = time.monotonic()
                    if direction is not None:
                        self.send_key_up(direction)
                    if self.wait_safe_lock_pick_active(time_out=2, settle_time=0.25):
                        self.wait_until(
                            lambda: not self.is_safe_lock_pick_active(),
                            time_out=10,
                            settle_time=0.5,
                        )
                        self.sleep(0.5, check_reward=False, scaled=False)
                    deadline += time.monotonic() - lock_pick_start
                    if direction is not None:
                        self.send_key_down(direction)
                self.next_frame()
        finally:
            if direction is not None and not hold:
                self.send_key_up(direction)
            if send_pick and pick_started:
                self.send_key_up("f")

    def wait_for_safe_loot(self, time_out=10, raise_timeout=False):
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            if self.has_safe_lock_prompt():
                self.wait_safe_lock_pick_active(time_out=2)
            if self.is_safe_lock_pick_active():
                self.wait_until(
                    lambda: not self.is_safe_lock_pick_active(),
                    time_out=10,
                    settle_time=0.5,
                )
                self.sleep(0.5, check_reward=False, scaled=False)
                return True
            self.next_frame()
        if raise_timeout:
            raise AbortException("timeout for wait_for_safe_loot")
        return False

    def has_extract_panel(self):
        return self._recognize_once("PinkPawHeist_CheckEvacuateOnce")

    def should_early_extract(self, exit_index):
        if exit_index is None:
            return False
        return bool(self.early_extract_exit.get(int(exit_index), False))

    def try_open_exit(self, direction=None, exit_index=None):
        if not self.wait_for_interac(time_out=4):
            raise AbortException("not found exit interaction")
        if direction is not None:
            self.send_key_up(direction)
            self.sleep(0.3, check_reward=False)
        ret = self.wait_until(
            self.has_extract_panel,
            pre_action=lambda: self.send_key("f", interval=0.7),
            time_out=2.5,
        )
        if ret:
            if self.should_early_extract(exit_index):
                self.log_round_info(f"Exit {exit_index} available, early extract")
                self._release_held_keys()
                self.ah.release_controls()
                if not self.exit_heist():
                    raise AbortException(f"early extract at exit {exit_index} failed")
                raise EarlyExtractException(f"early extracted at exit {exit_index}")
            self.sleep(0.3, check_reward=False)
            self.send_key("esc", interval=0.5)
            self.sleep(0.5, check_reward=False)
        return ret

    def walk_until_extract_panel(self, direction=None, time_out=10):
        if direction is not None:
            self.send_key_down(direction)
        try:
            return self.wait_until(
                self.has_extract_panel,
                pre_action=lambda: self.send_key("f", interval=0.25),
                time_out=time_out,
                raise_if_not_found=True,
            )
        finally:
            if direction is not None:
                self.send_key_up(direction)

    def clear_current_combat(self, fighter_mode="all_desc"):
        self.switch_to_fighter(check_switched=True, mode=fighter_mode)
        self.fight_until_no_monster(timeout_no_monster=10000, wait_for_monster=True)
        self.switch_to_runner(check_switched=True)

    def check_monster(self):
        image = self.ctx.tasker.controller.post_screencap().wait().get()
        result = self.ctx.run_recognition("PinkPawHeist_CheckMonsterOnce", image)
        return result is not None and getattr(result, "hit", False)

    def wait_monster(self, timeout=6000):
        deadline = time.monotonic() + timeout / 1000.0
        while time.monotonic() < deadline:
            if self.check_monster():
                return True
            self.sleep(0.2)
        return False

    def attack_cycle(self, times=3, loot=False):
        for _ in range(times):
            self.ah.run_task("PinkPawHeist_Core1_Attack_Space")
        if loot:
            self.send_key("f")

    def fight_until_no_monster(
        self,
        timeout_no_monster=10000,
        wait_for_monster=True,
        role_to_switch_back=None,
        loot=False,
        attack_cycles=3,
    ):
        if wait_for_monster and not self.wait_monster(timeout=timeout_no_monster):
            return False
        no_monster_start = None
        while True:
            if self.check_monster():
                no_monster_start = None
                self.attack_cycle(times=attack_cycles, loot=loot)
            else:
                now = time.monotonic()
                if no_monster_start is None:
                    no_monster_start = now
                elif now - no_monster_start >= timeout_no_monster / 1000.0:
                    break
                self.sleep(0.05)
        if role_to_switch_back:
            self.switch_to_key(role_to_switch_back)
        return True

    def switch_to_key(self, key):
        for _ in range(4):
            self.send_key(str(key))
            self.sleep(0.2)
        return str(key)

    def _send_current_switch_key(self):
        state = self._switch_state
        if state is None:
            return None
        key = state.current_key
        if state.role == self.ROLE_FIGHTER:
            self._current_fighter_key = key
        state.deadline = time.monotonic() + self.SWITCH_CHECK_DURATION
        self._next_switch_poll_at = time.monotonic() + 0.05
        self.send_key(key)
        return key

    def _clear_switch_state(self):
        self._switch_state = None
        self._next_switch_poll_at = 0.0

    def _handle_dead_switch_candidate(self, state: CharacterSwitchState):
        role = state.role
        key = state.current_key
        self.log_warning(f"{role} char {key} may be dead, try next")
        if role == self.ROLE_FIGHTER and key not in self._dead_fighter_keys:
            self._dead_fighter_keys.append(key)
        self.ensure_in_team()
        if not state.advance():
            self._clear_switch_state()
            raise AbortException(f"{role} {state.keys} dead or empty")
        self._send_current_switch_key()

    def _poll_character_switch(self):
        if self._switch_state is None or self._handling_switch_state:
            return
        now = time.monotonic()
        if now < self._next_switch_poll_at:
            return
        self._next_switch_poll_at = now + 0.1

        state = self._switch_state
        if now > state.deadline:
            self._clear_switch_state()
            return

        image = self._screencap()
        if image is not None and self._is_black_screen_in_image(image):
            state.deadline = max(
                state.deadline,
                time.monotonic() + SWITCH_BLACK_SCREEN_EXTENSION,
            )
            return
        if image is None or self._is_in_team_in_image(image):
            return

        self._handling_switch_state = True
        try:
            self._handle_dead_switch_candidate(state)
        finally:
            self._handling_switch_state = False

    def _wait_character_switch_success(self, role, key):
        last_key = str(key)
        retry_count = 0
        retry_key = last_key
        not_team_since = None
        old_handling = self._handling_switch_state
        self._handling_switch_state = True
        try:
            while self._switch_state is not None:
                state = self._switch_state
                last_key = state.current_key
                if retry_key != last_key:
                    retry_key = last_key
                    retry_count = 0
                now = time.monotonic()
                if now > state.deadline:
                    if retry_count < SWITCH_CONFIRM_RETRY_COUNT:
                        retry_count += 1
                        self.log_warning(
                            f"{role} switch to {last_key} not confirmed, retry {retry_count}"
                        )
                        self.send_key(
                            last_key,
                            action_name=f"switch_char_retry:{last_key}",
                            interval=-1,
                        )
                        state.deadline = time.monotonic() + SWITCH_CONFIRM_RETRY_WINDOW
                        not_team_since = None
                        continue
                    self.log_warning(f"{role} switch to {last_key} not confirmed")
                    self._clear_switch_state()
                    return last_key

                self.send_key(last_key, action_name="switch_char", interval=0.5)
                image = self._screencap()
                if image is not None and self.is_char_at_index(
                    int(last_key) - 1, image=image
                ):
                    self._clear_switch_state()
                    return last_key

                if image is not None and self._is_black_screen_in_image(image):
                    state.deadline = max(
                        state.deadline,
                        time.monotonic() + SWITCH_BLACK_SCREEN_EXTENSION,
                    )
                    not_team_since = None
                    self.sleep(
                        WAIT_UNTIL_POLL_INTERVAL,
                        check_reward=False,
                        scaled=False,
                    )
                    continue

                in_team = True if image is None else self._is_in_team_in_image(image)
                if in_team:
                    not_team_since = None
                else:
                    if not_team_since is None:
                        not_team_since = now
                    elif now - not_team_since >= SWITCH_DEAD_SETTLE:
                        self._handle_dead_switch_candidate(state)
                        not_team_since = None

                self.sleep(WAIT_UNTIL_POLL_INTERVAL, check_reward=False, scaled=False)
        finally:
            self._handling_switch_state = old_handling

        return last_key

    def _begin_character_switch(self, role, keys, check_switched=False):
        keys = [str(key) for key in keys]
        if not keys:
            raise AbortException(f"{role} {keys} dead or empty")
        self._switch_state = CharacterSwitchState(role=role, keys=keys)
        key = self._send_current_switch_key()
        if check_switched:
            return self._wait_character_switch_success(role, key)
        return key

    def switch_to_runner(self, check_switched=False):
        return self._begin_character_switch(
            self.ROLE_RUNNER, self.config.get(self.CONF_RUNNER, []), check_switched
        )

    def switch_to_avoider(self, check_switched=False):
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            self.log_info("no avoider")
            return None
        return self._begin_character_switch(self.ROLE_AVOIDER, keys, check_switched)

    def avoider_strategy_index(self):
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            return -1
        method_name = self.config.get(self.CONF_AVOID_MTH)
        if method_name not in self.avoid_methods:
            return 0
        return self.avoid_methods.index(method_name)

    def perform_avoidance_action(self):
        method_name = self.config.get(self.CONF_AVOID_MTH)
        if method_name == self.AVOID_METHOD_ATTACK:
            self.click(down_time=0.6)
            return
        self.send_key_down("w")
        self.sleep(0.1)
        self.send_key_down("lshift")
        self.sleep(1.0)
        self.send_key_up("lshift")
        self.sleep(0.1)
        self.send_key_up("w")

    def exit_heist(self):
        self.log_round_info("Confirm extract")
        self.sleep(1.0, check_reward=False, scaled=False)
        result = self.ah.run_task("PinkPawHeist_EvacuateOnce")
        if _is_hit(result):
            self.sleep(REWARD_OCR_DELAY_MS / 1000.0, check_reward=False, scaled=False)
            notify_pinkpaw_reward(self.ctx, success=True)
            self.sleep(POST_REWARD_DELAY_MS / 1000.0, check_reward=False, scaled=False)
            return True
        notify_pinkpaw_reward(self.ctx, success=False)
        return False

    def abort_heist(self):
        self.log_round_info("Abort and return to main")
        self.ah.release_controls()
        for _ in range(4):
            self.send_key("esc")
            self.sleep(1.0, check_reward=False, scaled=False)
        self.ah.run_task("PinkPawHeist_Once")
        self.sleep(5.0, check_reward=False, scaled=False)
        notify_pinkpaw_reward(self.ctx, success=False)

    def _release_held_keys(self):
        held = list(self._held_keys)
        self._held_keys.clear()
        for key in held:
            try:
                self.ah.key_up(key)
            except Exception as exc:
                self.log_error(f"release held key {key} failed", exc)
        self._quick_pick_active = False

    def run_path(self):
        self.goto_lg1()
        self.wait_team_ui_settle()
        # self.check_current_floor(1)
        self.lg1_wp1()
        self.lg1_wp2()
        self.lg1_wp3()
        self.lg1_wp4()
        idx = self.avoider_strategy_index()
        if idx == -1:
            self.lg1_wp5_avoid_combat_01()
        elif idx == 0:
            self.lg1_wp5_avoid_combat_02()
        elif idx == 1:
            self.lg1_wp5_avoid_combat_03()
        self.wait_team_ui_settle()
        # self.check_current_floor(2)
        self.lg2_wp1_to_exit1()
        self.lg2_wp1_remains()
        self.lg2_wp2_to_exit2()
        self.lg2_wp3_to_layzer_room()
        self.lg2_wp3_in_layzer_room()
        self.lg2_wp4()
        if self.exit_state[1]:
            self.lg2_wp4_to_exit1()
        elif self.exit_state[2]:
            self.lg2_wp4_to_exit2()
        else:
            self.lg2_wp4_to_exit3()

    def goto_lg1(self):
        self.log_round_info("寻路到LG1")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.81)
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key_down("lshift")
        self.sleep(0.16)
        self.send_key_up("lshift")
        self.sleep(2.68)
        self.send_key_down("d")
        self.sleep(2.55)
        self.send_key_up("d")
        self.sleep(0.37)
        self.wait_and_interact(direction="w", is_lock=True)
        self.send_key_down("w")
        self.sleep(0.25)

        self.send_key_down("f")
        start = time.time()
        while time.time() < start + 10:
            self.send_key("space", down_time=0.14, interval=0.25)
            if time.time() > start + 6.4 and self.find_interac():
                break
            self.next_frame()

        self.wait_lock_pick_active(settle_time=0.5)
        self.send_key_up("f")
        self.send_key_up("w")
        self.wait_until(lambda: not self.is_lock_pick_active_fast(), settle_time=0.5)
        if self.find_interac():
            self.goto_lg1_interrupted()
        self.sleep(0.01)

        self.send_key_down("w")
        self.sleep(0.2)
        self.sleep_send_key(0.2, key="lshift")
        self.send_key_down("d")
        self.sleep_send_key(0.5, key="lshift")
        self.sleep(0.5)
        self.send_key_up("d")
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep_send_key(0.5, key="lshift")
        self.sleep(0.5)
        self.send_key_up("w")
        self.sleep_send_key(3.5, interval=0.7, key="lshift")
        self.send_key_up("a")

        self.sleep(0.04)
        self.send_key_down("s")
        self.sleep(0.29)
        self.send_key("lshift")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.04)
        self.send_key_down("d")
        self.sleep(0.29)
        self.send_key("lshift")
        self.sleep(2.50)
        self.send_key_up("d")
        self.sleep(0.40)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.36)
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.14)
        self.send_key_down("f")  # start pick
        self.sleep(0.04)
        self.send_key_down("w")
        self.sleep(2.5)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_up("f")  # end pick
        self.sleep(0.13)
        self.send_key_down("s")
        self.sleep(0.14)
        self.send_key_up("s")
        self.sleep(0.20)
        self.clear_current_combat()
        self.send_key_down("f")
        self.sleep(0.5)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("lshift")
        self.sleep(0.10)
        self.send_key_up("lshift")
        self.sleep(1)
        self.send_key_up("f")
        self.sleep(0.1)
        self.send_key_down("d")
        self.sleep(1)
        self.send_key_up("w")
        self.sleep(1.5)
        self.send_key_up("d")
        self.sleep(0.35)
        self.send_key_down("a")
        self.sleep(0.88)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("s")
        self.sleep(0.43)
        self.send_key_down("lshift")
        self.sleep(0.13)
        self.send_key_up("lshift")
        self.sleep(1.4)
        self.send_key_down("a")
        self.sleep(0.53)
        self.send_key_up("a")
        self.sleep(1.64)
        self.wait_and_interact(direction="s")
        self.sleep(0.50)
        self.send_key_down("s")
        self.sleep(0.10)
        self.wait_and_interact(direction="s")

    def goto_lg1_interrupted(self):
        self.log_round_info("LG1开锁中断恢复")
        self.clear_current_combat()
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(3.41)
        self.send_key_up("d")
        self.sleep(0.32)
        self.send_key_down("a")
        self.sleep(1.16)
        self.send_key_up("a")
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(1.51)
        self.send_key_up("w")
        self.sleep(0.11)
        self.wait_and_interact(direction="w", is_lock=True)
        self.sleep(0.5)

    def lg1_wp1(self):
        self.log_round_info("LG1 WP1")
        self.sleep(0.75)
        self.send_key_down("w")
        self.sleep(9.06)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.71)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.01)
        self.wait_and_interact(direction="s", key_up_sleep=0)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.25)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.03)
        self.send_key_up("d")
        self.sleep(0.22)
        self.send_key_down("a")
        self.sleep(3.90)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(0.40)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.60)
        self.send_key_up("d")
        self.sleep(0.06)
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.21)
        self.send_key_up("d")
        self.sleep(0.12)

    def lg1_wp2(self):
        self.log_round_info("LG1 WP2")
        self.send_key_down("d")
        self.sleep(1.80)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.71)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.28)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.26)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(2.93)
        self.send_key_down("w")  # 过镭射1
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.44)
        self.start_interaction_watch()
        self.send_key_down("w")  # 过镭射2
        self.sleep(8.51)
        self.send_key_up("w")
        self.stop_interaction_watch()
        self.sleep(0.33)

    def lg1_wp3(self):
        self.log_round_info("LG1 WP3")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(2.13)
        self.send_key_up("a")
        self.sleep(0.52)
        self.send_key_down("s")
        self.sleep(1.32)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.20)
        self.send_key_up("w")
        self.sleep(0.31)
        self.send_key_down("a")
        self.sleep(1.50)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key_up("d")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("w")
        self.sleep(3.19)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(5.16)
        self.send_key_up("w")
        self.stop_interaction_watch()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.15)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(5.31)
        self.send_key_up("w")
        self.sleep(0.12)

    def lg1_wp4(self):
        self.log_round_info("LG1 WP4")
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.31)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.50)
        self.send_key_up("w")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_up("a")
        self.sleep(1.22)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(6.58)
        self.send_key_down("d")
        self.sleep(2.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(1.30)
        self.start_interaction_watch()
        self.send_key_down("d")
        self.sleep(2.10)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.65)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.48)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.14)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(1.41)
        self.send_key_down("w")
        self.sleep(0.81)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.47)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.11)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(3.38)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.loot_safes_while_walking(direction="s", time_out=2.37)
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(1.33)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(9.40)
        self.send_key_up("w")
        self.sleep(0.31)

    def lg1_wp5_avoid_combat_01(self):
        self.log_round_info("LG1 WP5避战路线1")
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(0.81)
        self.send_key_up("s")
        self.sleep(2.01)
        self.send_key_down("w")
        self.sleep(0.11)

        deadline = time.time() + 4.5
        while time.time() < deadline:
            self.send_key("lshift")
            self.sleep(0.51)

        self.wait_and_interact(direction="w", is_lock=True)
        self.sleep(0.11)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg1_wp5_avoid_combat_02(self):
        self.log_round_info("LG1 WP5避战路线2")
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.11)

        self.switch_to_avoider(check_switched=True)
        self.sleep(0.5)
        self.perform_avoidance_action()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(6.0)
        self.send_key_up("w")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.5)
        self.wait_and_interact(is_lock=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg1_wp5_avoid_combat_03(self):
        self.log_round_info("LG1 WP5避战路线3")
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.5)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.send_key_down("w")
        self.sleep(0.11)

        deadline = time.time() + 4.5
        while time.time() < deadline:
            self.send_key("lshift")
            self.sleep(0.51)

        self.send_key_up("w")
        self.sleep(0.2)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.5)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key_up("d")
        self.wait_and_interact(is_lock=True)
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.20)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg2_wp1_to_exit1(self):
        self.log_round_info("LG2 WP1尝试出口1")
        self.sleep(2.65)  # 2.65
        self.send_key_down("w")
        self.sleep(4.92)
        self.send_key_up("w")
        self.sleep(0.13)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(3.00)
        self.send_key("lshift")  # x0.6
        self.sleep(3.10)
        self.send_key_up("a")
        self.sleep(0.21)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("s")
        self.sleep(0.33)
        self.send_key_down("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.41)
        self.send_key_up("d")
        self.sleep(1.21)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.52)
        self.send_key_up("d")
        self.sleep(0.29)
        self.send_key_down("s")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.93)
        self.send_key_up("d")
        self.sleep(0.21)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.71)
        self.exit_state[1] = self.try_open_exit(direction="w", exit_index=1)

    def lg2_wp1_remains(self):
        self.log_round_info("LG2 WP1剩余路线")
        self.send_key_down("w")
        self.sleep(2.05)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep(0.90)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("w")
        self.sleep(3.00)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.06)
        self.send_key_up("w")
        self.sleep(0.11)
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(5.41)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.2)
        self.send_key("lshift")
        self.sleep(1.37)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.31)
        self.send_key_up("w")
        self.sleep(0.18)
        self.send_key_down("d")
        self.sleep(0.70)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("s")
        self.sleep(3.02)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.72)
        self.send_key_down("s")
        self.sleep(6.36)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.07)
        self.send_key_up("d")
        self.sleep(0.21)
        self.send_key_up("f")  # end pick
        self.send_key_down("w")
        self.sleep(1.81)

        self.send_key_down("space")
        self.send_key_down("f")  # start pick
        self.sleep(0.13)
        self.send_key_up("space")
        self.sleep(0.17)
        self.send_key_down("space")
        self.sleep(0.13)
        self.send_key_up("space")
        self.sleep(3.96)
        self.send_key_down("a")
        self.sleep(0.13)
        self.send_key_up("a")
        self.sleep(0.53)
        self.send_key_down("d")
        self.sleep(0.13)
        self.send_key_up("d")
        self.sleep(4.64)
        self.send_key_down("d")
        self.sleep(1.31)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.14)
        self.send_key_down("s")
        self.sleep(0.22)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.71)
        self.send_key_up("w")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.72)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.90)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.02)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.30)
        self.send_key_down("a")
        self.sleep(0.68)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.26)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.01)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.29)
        self.send_key_down("a")
        self.sleep(0.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(7.12)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.44)
        self.send_key_up("d")
        self.sleep(0.91)
        self.send_key_up("w")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.72)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.61)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)

    def lg2_wp2_to_exit2(self):
        self.log_round_info("LG2 WP2尝试出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.21)
        self.send_key_down("space")
        self.sleep(0.06)
        self.send_key_up("space")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(1.70)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key("lshift")
        self.sleep(2.64)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.96)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.15)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.61)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.20)
        self.send_key_down("a")
        self.sleep(0.72)
        self.send_key_up("a")
        self.sleep(1.26)
        self.send_key_down("w")
        self.sleep(2.60)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.31)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.63)  # 4.03
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.75)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.56)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(1.57)
        self.exit_state[2] = self.try_open_exit(direction="w", exit_index=2)
        self.sleep(0.40)

    def lg2_wp3_to_layzer_room(self):
        self.log_round_info("LG2 WP3前往镭射房")
        self.send_key_down("a")
        self.sleep(3.03)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.55)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.56)
        self.send_key_up("s")
        self.sleep(1.18)
        self.send_key_down("a")
        self.sleep(2.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.77)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.60)
        self.send_key_up("a")
        self.sleep(0.29)
        self.send_key_down("d")
        self.sleep(1.31)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.76)
        self.send_key_up("s")
        self.sleep(0.30)
        self.send_key_down("a")
        self.sleep(0.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.97)
        self.send_key_up("s")

    def lg2_wp3_in_layzer_room(self):
        self.log_round_info("LG2 WP3镭射房")
        self.send_key_down("d")
        self.sleep(0.36)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.wait_for_safe_loot(time_out=1.5)
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.46)
        self.send_key_up("a")

        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.38)
        self.send_key_up("w")
        self.sleep(0.36)
        self.send_key_down("a")
        self.sleep(2.01)
        self.send_key_down("s")
        self.sleep(0.23)
        self.send_key_up("s")
        self.sleep(1.01)
        self.send_key_down("s")
        self.sleep(0.23)
        self.send_key_up("s")
        self.sleep(1.01)
        self.send_key_down("w")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.27)
        self.send_key_up("w")

        self.sleep(0.37)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.30)
        self.send_key_down("s")
        self.sleep(0.35)
        self.send_key_up("s")

        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.41)
        self.send_key_up("a")
        self.sleep(0.05)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.30)
        self.send_key_up("d")
        self.sleep(0.54)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.16)
        self.send_key_down("a")
        self.sleep(2.01)
        self.send_key_up("a")
        self.sleep(0.16)
        self.send_key_down("d")
        self.sleep(0.91)
        self.send_key_up("d")
        self.sleep(0.13)
        self.send_key_down("a")
        self.sleep(0.13)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.33)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.40)
        self.send_key_down("s")
        self.sleep(0.24)
        self.send_key_down("space")
        self.sleep(0.07)
        self.send_key_up("space")
        self.sleep(1.21)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.31)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.74)
        self.send_key_up("a")
        self.sleep(0.80)
        self.send_key_down("d")
        self.sleep(0.70)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.52)
        self.send_key_up("s")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.wait_for_safe_loot(raise_timeout=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.90)
        self.send_key_up("d")
        self.sleep(1.32)
        self.send_key_up("w")
        self.sleep(0.92)
        self.send_key_down("s")
        self.sleep(0.08)
        self.send_key_down("d")
        self.sleep(1.36)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.51)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.17)
        self.send_key_up("s")
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(2.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.66)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.54)
        self.send_key_up("a")
        self.sleep(0.13)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.39)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_down("a")
        self.sleep(3.01)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.51)
        self.send_key_up("w")
        self.sleep(0.13)
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.31)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.73)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.43)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(0.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.22)
        self.send_key_down("s")
        self.sleep(0.03)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(1.21)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.31)
        self.send_key_down("s")
        self.sleep(0.40)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_down("a")
        self.sleep(1.41)
        self.send_key_up("a")
        self.sleep(0.11)

    def lg2_wp4(self):
        self.log_round_info("LG2 WP4")
        self.send_key_down("w")
        self.sleep(4.40)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.20)
        self.send_key_down("lshift")
        self.sleep(0.20)
        self.send_key_up("lshift")
        self.sleep(1.5)
        self.send_key_up("a")
        self.sleep(0.01)
        self.send_key_up("f")  # end pick

    def lg2_wp4_to_exit1(self):
        self.log_round_info("LG2 WP4前往出口1")
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep(0.17)
        self.send_key_down("lshift")
        self.sleep(0.14)
        self.send_key_up("lshift")
        self.sleep(4.69)
        self.send_key_up("a")
        self.sleep(0.41)
        self.send_key_down("d")
        self.sleep(0.31)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_down("lshift")
        self.sleep(0.23)
        self.send_key_up("lshift")
        self.sleep(4.55)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)

        deadline = time.time() + 1.29
        while time.time() < deadline:
            self.send_key("space")
            self.sleep(0.25)

        self.sleep(1.21)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.walk_until_extract_panel(direction="w")

    def lg2_wp4_to_exit2(self):
        self.log_round_info("LG2 WP4前往出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(0.11)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(1.10)
        self.send_key_down("d")
        self.sleep(0.90)
        self.send_key_up("w")
        self.sleep(2.30)
        self.send_key_down("s")
        self.sleep(1.01)
        self.send_key_up("s")
        self.sleep(0.21)
        self.send_key_down("w")
        self.sleep(0.74)
        self.send_key_up("w")
        self.sleep(4.61)
        self.send_key_up("d")
        self.sleep(0.41)
        self.send_key_down("s")
        self.sleep(1.00)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.00)
        self.send_key_down("w")
        self.sleep(1.00)
        self.send_key_up("w")
        self.walk_until_extract_panel(direction="d")

    def lg2_wp4_to_exit3(self):
        self.log_round_info("LG2 WP4前往出口3")
        self.send_key_down("w")
        self.sleep(0.14)
        self.send_key_down("lshift")
        self.sleep(0.13)
        self.send_key_up("lshift")
        self.sleep(2.70)
        self.send_key_down("a")
        self.sleep(1.98)
        self.send_key_up("a")
        self.wait_and_interact(direction="w", is_lock=True, time_out=9)
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.05)
        self.send_key_down("lshift")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.05)
        self.send_key_up("lshift")
        self.sleep(1.76)
        self.send_key_up("w")
        self.sleep(0.26)
        self.send_key_up("d")
        self.sleep(0.10)
        self.walk_until_extract_panel(direction="d")

    def run_path(self):
        idx = self.avoider_strategy_index()
        if idx == -1:
            self.log_round_info("没有配置避战角色，全程使用原始线路（路线A）")
            self.goto_lg1()
        elif idx == 0:
            self.log_round_info("配置避战角色狗哥，使用早雾避战（路线B）")
            self.goto_lg1_skip_Sakiri()
        elif idx == 1:
            self.log_round_info("配置避战角色浔，使用浔避战（路线B）")
            self.goto_lg1_skip_Hotori()
        self.wait_team_ui_settle()
        # if not self.check_current_floor_str("办公"):
        #     self.check_current_floor(1)
        self.switch_to_runner(check_switched=True)
        self.lg1_wp1_safer()
        self.lg1_wp2()
        self.lg1_wp3()
        if idx == -1:
            self.lg1_wp4()
            self.lg1_wp5_avoid_combat_01()
        elif idx == 0:
            self.lg1_wp4_buster()
            self.lg1_wp5_buster()
        elif idx == 1:
            self.lg1_wp4()
            self.lg1_wp5_avoid_combat_03()
        self.wait_team_ui_settle()
        # if not self.check_current_floor_str("藏品"):
        #     self.check_current_floor(2)
        self.lg2_wp1_to_exit1()  # self.lg2_wp1_to_exit1_safer(False)
        self.lg2_wp1_remains()
        self.lg2_wp2_to_exit2_safer()
        self.lg2_wp3_to_layzer_room()
        self.lg2_wp3_in_layzer_room()
        self.lg2_wp4()
        if self.exit_state[1]:
            self.lg2_wp4_to_exit1()
        elif self.exit_state[2]:
            self.lg2_wp4_to_exit2()
        else:
            self.lg2_wp4_to_exit3()

    def goto_lg1_skip_Sakiri(self):
        self.log_round_info("早雾、大厅前往LG1")
        self.sleep(0.30)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key_up("w")
        self.sleep(0.20)
        self.send_key_down("d")
        self.sleep(0.57)
        self.send_key("lshift", down_time=0.32)
        self.sleep(0.57)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.40)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.60)
        self.switch_to_avoider(check_switched=True)  # 切到狗哥潜行避免碰到怪改变路径
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.2)
        self.send_key_down("w")
        self.sleep(0.15)
        self.send_key_down("lshift")
        self.sleep(0.24)
        self.send_key("d", down_time=0.30)
        self.sleep(1.28)
        self.send_key_up("lshift")
        self.sleep(0.64)
        self.send_key("d", down_time=0.12)
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(1.14)
        self.send_key_up("w")
        self.sleep(0.10)
        self.switch_to_fighter(check_switched=True, mode=1)  # 切到早雾控怪
        self.sleep(0.10)
        self.send_key("a", down_time=0.20)
        self.sleep(0.10)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        if self.find_interac():
            self.send_key("s", down_time=0.10)
            self.sleep(0.25)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key("e", down_time=2.40)
            self.send_key_down("w")
            self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        self.switch_to_avoider(check_switched=True)  # 切到狗哥潜行避免碰到怪改变路径
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key("d", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.42)
        self.sleep(0.76)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key("a", down_time=3.80)
        self.sleep(0.10)
        self.send_key("a", down_time=0.10)
        self.sleep(0.10)
        self.send_key("w", down_time=0.20)
        self.sleep(0.10)
        self.send_key("w", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=1.14)
        self.sleep(0.10)
        self.send_key("d", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=1.60)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.20)
        self.click(0.50, 0.50, key="middle", down_time=0.15)
        self.sleep(0.30)
        self.send_key_down("w")
        self.sleep(0.20)
        self.send_key("lshift", down_time=1.25)
        self.sleep(1.80)
        self.send_key("d", down_time=0.32)
        self.sleep(0.90)
        self.send_key_down("d")
        self.sleep(0.80)
        self.send_key_up("d")
        self.send_key_up("w")
        self.switch_to_fighter(check_switched=True, mode=1)  # 切到早雾控怪
        self.sleep(0.24)
        self.send_key("s", down_time=0.10)
        self.sleep(1.14)
        self.send_key("e", down_time=2.60)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.14)
        self.send_key_down("d")
        self.wait_and_interact(direction="d", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_up("w")
        if self.wait_ocr(
            x=0.60,
            y=0.52,
            to_x=0.70,
            to_y=0.57,
            match=re.compile("开门"),
            time_out=1.14,
        ):
            self.sleep(0.10)
            self.send_key("f", down_time=0.10)
            self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("a", down_time=0.36)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def goto_lg1_skip_Hotori(self):
        self.log_round_info("浔、大厅前往LG1")
        self.sleep(0.30)
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.64)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.64)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.64)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.64)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(0.64)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.64)
        self.send_key_up("d")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.60)
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.2)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key_up("d")
        self.sleep(0.24)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.64)
        self.send_key("space", down_time=0.24)
        self.sleep(0.84)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.84)
        self.send_key_up("w")
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        if self.find_interac():
            self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key("d", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.42)
        self.sleep(0.76)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(0.20)
        self.send_key("lshift", down_time=0.20)
        self.sleep(0.20)
        self.send_key("lshift", down_time=0.20)
        self.sleep(0.20)
        self.send_key("lshift", down_time=0.20)
        self.sleep(1.20)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key("a", down_time=0.10)
        self.sleep(0.10)
        self.send_key("w", down_time=0.20)
        self.sleep(0.10)
        self.send_key("w", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=1.16)
        self.sleep(0.10)
        self.send_key("d", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=1.82)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.20)
        self.click(0.50, 0.50, key="middle", down_time=0.15)
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(1.14)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.42)
        self.send_key_down("d")
        self.sleep(0.42)
        self.send_key_up("d")
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.76)
        self.send_key_up("w")
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("a", down_time=0.42)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def lobby_open_door_check(self, check_time=3):
        open_door = False
        open_loop = 0
        while not open_door and open_loop < check_time:
            if self.wait_ocr(
                x=0.60,
                y=0.52,
                to_x=0.70,
                to_y=0.57,
                match=re.compile("开门"),
                time_out=1.14,
            ):
                open_door = True
            else:
                self.sleep(0.10)
                self.send_key("f", down_time=0.10)
                self.sleep(0.20)
                open_loop += 1
        return open_door

    def lg1_wp1_safer(self):
        self.log_round_info("LG1 WP1 Safer")
        self.switch_to_runner(check_switched=True)  # 确认切到薄荷跑图
        self.sleep(0.20)
        self.send_key("w", down_time=9.08)
        self.sleep(0.10)
        self.send_key("d", down_time=1.72)
        self.sleep(0.10)
        self.send_key("s", down_time=1.00)
        self.sleep(0.10)
        self.send_key(
            "f", down_time=0.10
        )  # 这里没必要上检测，门口不安全，停太久可能会被蚊子扫
        self.sleep(0.10)
        self.send_key("f", down_time=0.10)
        self.sleep(0.20)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.25)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.03)
        self.send_key_up("d")
        self.sleep(0.22)
        self.send_key_down("a")
        self.sleep(3.90)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(0.40)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.60)
        self.send_key_up("d")
        self.sleep(0.06)
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.21)
        self.send_key_up("d")
        self.sleep(0.12)

    def lg1_wp4_buster(self):
        self.log_round_info("LG1 WP4 bUSTER")
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.31)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.50)
        self.send_key_up("w")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_up("a")
        self.sleep(1.22)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(6.58)
        self.send_key_down("d")
        self.sleep(2.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(1.25)
        self.start_interaction_watch()
        self.send_key_down("d")
        self.sleep(2.10)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.65)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.48)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.14)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(1.41)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(0.81)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.47)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.38)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.loot_safes_while_walking(direction="s", time_out=2.37)
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(1.33)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(7.60)
        self.send_key_up("w")

    def lg1_wp5_buster(self):
        self.log_round_info("LG1 WP5 Buster 开始避战路线")
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.50)
        self.perform_avoidance_action()
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(6.00)
        self.send_key_up("w")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.32)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key_up("d")
        self.wait_and_interact(is_lock=True)
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(0.15)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.10)
        self.wait_and_interact(direction="w")

    def lg2_wp2_to_exit2_safer(self):
        self.log_round_info("LG2 WP2 Safer 尝试出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.21)
        self.send_key_down("space")
        self.sleep(0.10)
        self.send_key_up("space")
        self.sleep(0.80)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_up("f")  # end pick
        self.send_key_down("w")
        self.sleep(1.80)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.80)
        self.send_key("lshift", down_time=0.10)
        self.sleep(2.00)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.96)
        self.send_key_up("w")
        self.switch_to_runner()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.61)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.20)
        self.send_key_down("a")
        self.sleep(0.72)
        self.send_key_up("a")
        self.sleep(1.26)
        self.send_key_down("w")
        self.sleep(2.60)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.31)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(4.03)  # 4.03
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.85)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.56)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(1.57)
        self.exit_state[2] = self.try_open_exit(direction="w", exit_index=2)
        self.sleep(0.40)

    def check_current_floor_str(self, floor_str):
        ret = self.wait_ocr(
            0.04, 0.23, 0.17, 0.28, match=re.compile(floor_str), time_out=5
        )
        if ret:
            return True

    def switch_to_fighter(self, check_switched=False, mode="all_desc"):
        """切换到可用战斗角色。
        `mode` 调度策略（配置重新从小到大排序后）：
        - "all_desc": [默认]按键位从大到小完整尝试（如 ["4", "1"]）
        - "all_asc" : 按键位从小到大完整尝试（如 ["1", "4"]）
        -     1     : 只切当前最小的那个键位
        -    -1     : 只切当前最大的那个键位
        -     n     : 只切重新排序后的【第 n 个】角色
        """
        config_keys = list(self.config.get(self.CONF_FIGHTER, []))
        if not config_keys:
            dead_keys = set(self._dead_fighter_keys)
            config_keys = [item for item in config_keys if item not in dead_keys]
            return self._begin_character_switch(
                self.ROLE_FIGHTER, config_keys, check_switched
            )
        sorted_keys = sorted(config_keys, key=int)
        if mode == "all_asc":
            keys = sorted_keys
        elif mode == "all_desc":
            keys = sorted_keys[::-1]
        elif isinstance(mode, int):
            if mode == -1:
                keys = [sorted_keys[-1]]
            else:
                idx = mode - 1
                if 0 <= idx < len(sorted_keys):
                    keys = [sorted_keys[idx]]
                else:
                    self.log_error(
                        f"切人位置越界！配置排序后只有 {len(sorted_keys)} 个人，你请求切第 {mode} 个，自动切最后一个。"
                    )
                    keys = [sorted_keys[-1]]
        else:
            keys = sorted_keys[::-1]
        dead_keys = set(self._dead_fighter_keys)
        keys = [item for item in keys if item not in dead_keys]
        return self._begin_character_switch(self.ROLE_FIGHTER, keys, check_switched)


@AgentServer.custom_action("PinkPawHeistScheme3Action")
class PinkPawHeistScheme3Action(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = _parse_custom_action_param(argv)
        if PinkPawHeistCore3Path.CONF_AVOID_MTH not in params:
            node_name = getattr(argv, "node_name", "")
            if node_name.endswith("_Attack"):
                params[PinkPawHeistCore3Path.CONF_AVOID_MTH] = (
                    PinkPawHeistCore3Path.AVOID_METHOD_ATTACK
                )
            elif node_name.endswith("_Dash"):
                params[PinkPawHeistCore3Path.CONF_AVOID_MTH] = (
                    PinkPawHeistCore3Path.AVOID_METHOD_DASH
                )
        path = PinkPawHeistCore3Path(context, params=params)
        try:
            path.log_round_info(
                f"Start copied OK-NTE route B, method {path.config[path.CONF_AVOID_MTH]}, timing x{path.route_timing_scale:.2f}"
            )
            path.run_path()
            path._release_held_keys()
            path.ah.release_controls()
            path.exit_heist()
            return CustomAction.RunResult(success=True)
        except TaskerStoppedException as exc:
            print(f"[PinkPawHeist/Core3] stopped by tasker: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)
        except EarlyExtractException as exc:
            print(f"[PinkPawHeist/Core3] {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=True)
        except AbortException as exc:
            print(f"[PinkPawHeist/Core3] route aborted: {exc}")
            path._release_held_keys()
            path.abort_heist()
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            print(f"[PinkPawHeist/Core3] route failed: {exc}")
            path._release_held_keys()
            path.abort_heist()
            return CustomAction.RunResult(success=True)
