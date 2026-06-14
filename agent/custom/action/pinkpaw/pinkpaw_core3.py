from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path
from ctypes import wintypes

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

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
    "m": 0x4D,
    "f5": 0x74,
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
DEFAULT_DIRECT_INPUT = True
MIN_ROUTE_TIMING_SCALE = 0.25
MAX_ROUTE_TIMING_SCALE = 1.2
MIN_INTERACTION_PAUSE = 0.0
MAX_INTERACTION_PAUSE = 1.0
MAX_ROUTE_SLEEP_ADJUST = 0.25
ROUTE_SLEEP_ADJUST_RATIO_CAP = 0.08
ROUTE_SLEEP_BUSY_WAIT = 0.02
ROUTE_SLEEP_POLL_INTERVAL = 0.005
ROUTE_REWARD_CHECK_MIN_SLEEP = 0.5
REWARD_CHECK_INTERVAL = 1.0
WAIT_UNTIL_POLL_INTERVAL = 0.02
INTERAC_OCR_FALLBACK_INTERVAL = 1.0
FOCUS_LOG_NODE = "_PINKPAW_CORE3_FOCUS_"
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
DIRECT_KEY_TAP_DURATION = 0.01
DIRECT_QUICK_PICK_TAP_DURATION = 0.002
DIRECT_ACTION_KEY_MIN_TAP_DURATION = 0.05
DIRECT_ACTION_KEYS = {"space", "lshift", "shift"}
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
        """返回当前正在尝试切换的角色按键。"""
        return self.keys[self.index]

    def advance(self):
        """把角色切换候选推进到下一个按键，并返回是否还有候选可试。"""
        self.index += 1
        return self.index < len(self.keys)


def _is_hit(result) -> bool:
    """兼容 MAA 不同返回结构，统一判断识别或任务是否成功命中。"""
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
    """把配置里的按键名规范成小写字符串，便于查虚拟键码。"""
    return str(key).lower()


def _normalize_key_sequence(value) -> list[str]:
    """Normalize one key or a sequence of keys into a de-duplicated key list."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if len(text) > 1 and all(char.lower() in {"w", "a", "s", "d"} for char in text):
            keys = list(text)
        else:
            keys = text.replace("+", " ").replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        keys = []
        for item in value:
            keys.extend(_normalize_key_sequence(item))
    else:
        keys = [str(value)]

    result = []
    for key in keys:
        normalized = _norm_key(key)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _parse_custom_action_param(argv: CustomAction.RunArg) -> dict:
    """解析节点传入的 custom_action_param，非法或空值时返回空配置。"""
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
    """解析路线时间微调倍率，并限制在允许范围内。"""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = DEFAULT_ROUTE_TIMING_SCALE
    return max(MIN_ROUTE_TIMING_SCALE, min(MAX_ROUTE_TIMING_SCALE, scale))


def _parse_bool(value, default=False) -> bool:
    """把开关配置解析为布尔值，兼容字符串、数字和 bool。"""
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
    """解析交互前停顿配置，避免停顿过短或过长。"""
    try:
        pause = float(value)
    except (TypeError, ValueError):
        pause = DEFAULT_INTERACTION_PAUSE
    return max(MIN_INTERACTION_PAUSE, min(MAX_INTERACTION_PAUSE, pause))


def _get_fast_image_dir():
    """按资源目录约定查找 PinkPawHeist 的快速识别模板目录。"""
    global _FAST_IMAGE_DIR
    if _FAST_IMAGE_DIR is not None:
        return _FAST_IMAGE_DIR
    project_root = Path(__file__).resolve().parents[4]
    candidates = [
        Path.cwd() / "assets" / "resource" / "base" / "image" / "PinkPawHeist",
        Path.cwd() / "resource" / "base" / "image" / "PinkPawHeist",
        project_root / "assets" / "resource" / "base" / "image" / "PinkPawHeist",
        project_root / "resource" / "base" / "image" / "PinkPawHeist",
    ]
    for candidate in candidates:
        if candidate.exists():
            _FAST_IMAGE_DIR = candidate
            return candidate
    _FAST_IMAGE_DIR = candidates[0]
    return _FAST_IMAGE_DIR


def _load_fast_template(name):
    """读取并缓存 OpenCV 模板图，供快速模板匹配复用。"""
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
    """把 MAA 截图转换为 OpenCV 使用的 BGR 三通道图。"""
    if np is None or not isinstance(image, np.ndarray):
        return None
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    return image[:, :, :3]


def _crop_roi(image, roi):
    """按给定坐标裁剪截图区域，并自动处理越界。"""
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
    """把以 1280x720 为基准的 ROI 缩放到当前截图尺寸。"""
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
    """在本地用 OpenCV 做颜色点数量检测，替代对应颜色识别节点。"""
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
    """在本地用 OpenCV 做模板匹配，减少频繁调用 MAA 节点的延迟。"""
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
    """根据节点名选择本地快速识别实现；不支持时交回 MAA 节点识别。"""
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


ULONG_PTR = (
    ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


class DirectInputSender:
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MAPVK_VK_TO_VSC = 0
    MOUSE_FLAGS = {
        "left": (0x0002, 0x0004),
        "right": (0x0008, 0x0010),
        "middle": (0x0020, 0x0040),
    }

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.available = False
        self.user32 = None
        if not self.enabled:
            return
        try:
            self.user32 = ctypes.windll.user32
            self.user32.SendInput.argtypes = [
                wintypes.UINT,
                ctypes.POINTER(_INPUT),
                ctypes.c_int,
            ]
            self.user32.SendInput.restype = wintypes.UINT
            self.available = True
        except Exception as exc:
            print(f"[PinkPawHeist/Core3][WARN] direct input unavailable: {exc}")

    def _send(self, input_obj):
        if not self.available or self.user32 is None:
            return False
        sent = self.user32.SendInput(
            1, ctypes.byref(input_obj), ctypes.sizeof(input_obj)
        )
        return sent == 1

    def _keyboard_input(self, vk, is_up=False):
        scan = int(self.user32.MapVirtualKeyW(int(vk), self.MAPVK_VK_TO_VSC))
        flags = self.KEYEVENTF_KEYUP if is_up else 0
        w_vk = int(vk)
        if scan:
            flags |= self.KEYEVENTF_SCANCODE
            w_vk = 0
        input_obj = _INPUT()
        input_obj.type = self.INPUT_KEYBOARD
        input_obj.u.ki = _KEYBDINPUT(
            wVk=w_vk,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        return input_obj

    def key_down(self, vk):
        return self._send(self._keyboard_input(vk, is_up=False))

    def key_up(self, vk):
        return self._send(self._keyboard_input(vk, is_up=True))

    def click_key(self, vk, duration=DIRECT_KEY_TAP_DURATION):
        if not self.key_down(vk):
            return False
        released = False
        try:
            time.sleep(max(float(duration), 0.0))
            released = self.key_up(vk)
            return released
        finally:
            if not released:
                self.key_up(vk)

    def _mouse_input(self, flags):
        input_obj = _INPUT()
        input_obj.type = self.INPUT_MOUSE
        input_obj.u.mi = _MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        return input_obj

    def mouse_down(self, key="left"):
        flags = self.MOUSE_FLAGS.get(key, self.MOUSE_FLAGS["left"])[0]
        return self._send(self._mouse_input(flags))

    def mouse_up(self, key="left"):
        flags = self.MOUSE_FLAGS.get(key, self.MOUSE_FLAGS["left"])[1]
        return self._send(self._mouse_input(flags))


class Core3ActionHelper:
    def __init__(self, ctx: Context, direct_input=True):
        """保存 MAA 上下文，并初始化鼠标当前位置缓存。"""
        self.ctx = ctx
        self.mx, self.my = DEFAULT_WIDTH // 2, DEFAULT_HEIGHT // 2
        self.direct_input = DirectInputSender(enabled=direct_input)

    @property
    def controller(self):
        """取得当前 tasker 的控制器，用于直接发送按键、鼠标和截图请求。"""
        return getattr(getattr(self.ctx, "tasker", None), "controller", None)

    def is_stopping(self) -> bool:
        """检查 MAA tasker 是否正在停止任务。"""
        tasker = getattr(self.ctx, "tasker", None)
        if tasker is None:
            return False
        stopping = getattr(tasker, "stopping", False)
        if callable(stopping):
            stopping = stopping()
        return bool(stopping)

    def raise_if_stopped(self):
        """任务停止时抛出专用异常，打断正在执行的路线。"""
        if self.is_stopping():
            raise TaskerStoppedException(
                "PinkPawHeistScheme3Action stopped by Maa tasker."
            )

    def run_task(self, task_name, pipeline_override=None):
        """运行一个 MAA pipeline 节点，并在调用前后检查停止状态。"""
        self.raise_if_stopped()
        if pipeline_override is None:
            result = self.ctx.run_task(task_name)
        else:
            result = self.ctx.run_task(task_name, pipeline_override=pipeline_override)
        self.raise_if_stopped()
        return result

    def _call_key(self, node_type, key_str, extra=None):
        """发送 KeyDown、KeyUp 或 ClickKey；有控制器时走低延迟直发，否则临时跑节点。"""
        if node_type != "KeyUp":
            self.raise_if_stopped()
        vk = VK.get(_norm_key(key_str))
        if vk is None:
            return False
        direct = self.direct_input
        key_name = _norm_key(key_str)
        if direct.available:
            if node_type == "KeyDown" and direct.key_down(vk):
                if node_type != "KeyUp":
                    self.raise_if_stopped()
                return True
            if node_type == "KeyUp" and direct.key_up(vk):
                return True
            if node_type == "ClickKey":
                duration = DIRECT_KEY_TAP_DURATION
                if extra and "direct_duration" in extra:
                    duration = float(extra["direct_duration"])
                if direct.click_key(vk, duration=duration):
                    self.raise_if_stopped()
                    return True
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
            param.update(
                {key: value for key, value in extra.items() if key != "direct_duration"}
            )
        node_name = f"PinkPawHeist_{node_type}"
        override = {node_name: {"action": {"type": node_type, "param": param}}}
        ret = self.ctx.run_task(node_name, pipeline_override=override) is not None
        if node_type != "KeyUp":
            self.raise_if_stopped()
        return ret

    def click_key(self, key_str, duration=None):
        """发送一次按键点击。"""
        key = _norm_key(key_str)
        if duration is None:
            duration = (
                DIRECT_QUICK_PICK_TAP_DURATION
                if key == "f"
                else DIRECT_KEY_TAP_DURATION
            )
        extra = None
        if duration is not None:
            extra = {"direct_duration": max(float(duration), 0.0)}
        return self._call_key("ClickKey", key_str, extra=extra)

    def key_down(self, key_str):
        """发送按键按下事件。"""
        return self._call_key("KeyDown", key_str)

    def key_up(self, key_str):
        """发送按键抬起事件。"""
        return self._call_key("KeyUp", key_str)

    def move_to(self, x, y, duration_ms=None):
        """把鼠标移动到指定坐标，并维护内部鼠标位置缓存。"""
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
        """点击指定坐标；控制器可用时直接点击，否则走 MAA Click 节点。"""
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

    def focus_window(self, x=None, y=None):
        """Use a controller click to bring the game window to foreground."""
        self.raise_if_stopped()
        px = DEFAULT_WIDTH // 2 if x is None else int(x)
        py = DEFAULT_HEIGHT // 2 if y is None else int(y)
        controller = self.controller
        if controller is not None and hasattr(controller, "post_click"):
            ret = controller.post_click(px, py)
            if hasattr(ret, "wait"):
                ret.wait()
            self.mx, self.my = px, py
            self.raise_if_stopped()
            return True
        return self.click(px, py)

    def mouse_down(self, key="left"):
        """发送鼠标按下事件，主要用于长按攻击或鼠标键操作。"""
        if self.direct_input.mouse_down(key=key):
            return
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_down(vk)

    def mouse_up(self, key="left"):
        """发送鼠标抬起事件，配合 mouse_down 结束长按。"""
        if self.direct_input.mouse_up(key=key):
            return
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_up(vk)

    def release_controls(self):
        """释放脚本可能按住的移动键、交互键和鼠标键，防止异常后继续输入。"""
        for key in ("w", "a", "s", "d", "e", "f", "space", "lshift"):
            try:
                self.key_up(key)
            except Exception as exc:
                print(f"[PinkPawHeist/Core3] failed to release {key}: {exc}")
        for key in MOUSE_VK:
            try:
                self.direct_input.mouse_up(key)
            except Exception as exc:
                print(
                    f"[PinkPawHeist/Core3] failed to release direct mouse {key}: {exc}"
                )
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
        """读取 Core3 配置，初始化路线状态、切人状态、拾取状态和撤离策略。"""
        params = params or {}
        self.ctx = ctx
        direct_input = _parse_bool(params.get("direct_input"), DEFAULT_DIRECT_INPUT)
        self.ah = Core3ActionHelper(ctx, direct_input=direct_input)
        self.exit_state = {1: False, 2: False, 3: False, 4: False}
        self.avoid_methods = [self.AVOID_METHOD_DASH, self.AVOID_METHOD_ATTACK]
        avoid_method = params.get(self.CONF_AVOID_MTH, self.AVOID_METHOD_DASH)
        if avoid_method not in self.avoid_methods:
            self.log_warning(f"unknown avoid_method {avoid_method!r}, fallback to dash")
            avoid_method = self.AVOID_METHOD_DASH
        self.route_timing_scale = _parse_timing_scale(
            params.get("timing_scale", DEFAULT_ROUTE_TIMING_SCALE)
        )
        self.interaction_pause = _parse_interaction_pause(
            params.get("interaction_pause", DEFAULT_INTERACTION_PAUSE)
        )
        self.early_extract_exit = {
            1: _parse_bool(params.get(self.CONF_EARLY_EXTRACT_EXIT1), False),
            2: _parse_bool(params.get(self.CONF_EARLY_EXTRACT_EXIT2), False),
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
        """输出 Core3 普通日志。"""
        print("[PinkPawHeist/Core3]", *args)

    def log_warning(self, *args):
        """输出 Core3 警告日志。"""
        print("[PinkPawHeist/Core3][WARN]", *args)

    def log_error(self, *args):
        """输出 Core3 错误日志。"""
        print("[PinkPawHeist/Core3][ERROR]", *args)

    def log_round_info(self, message):
        # self.log_info(f"{self._round_label}: {message}")
        """把路线阶段信息写到前端日志/Toast，便于观察当前跑到哪里。"""
        self._log_to_frontend(str(message))

    def _log_to_frontend(self, message: str):
        """通过临时 focus 节点把 Core3 日志显示到 MAA 前端。"""
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
        """按动作名做节流，避免同一个按键或点击在短时间内重复触发。"""
        if interval is None or interval < 0:
            return True
        now = time.monotonic()
        last = self._last_action_at.get(name, 0.0)
        if now - last < interval:
            return False
        self._last_action_at[name] = now
        return True

    def _poll_quick_pick(self):
        """自动拾取/撬锁时按固定频率点 F，并用发送完成时间避免连点堆积。"""
        if not self._quick_pick_active:
            return
        now = time.monotonic()
        if now < self._quick_pick_ready_at or now < self._next_quick_pick_at:
            return
        self.ah.click_key("f")
        self._next_quick_pick_at = now + self.QUICK_PICK_INTERVAL

    def _has_timing_sensitive_key_held(self) -> bool:
        """判断当前是否按着移动、冲刺、跳跃等会影响走位精度的键。"""
        return bool(self._held_keys & TIMING_SENSITIVE_KEYS)

    def _check_still_in_heist(self):
        """低频检测本局收益 UI，判断脚本是否仍在粉爪局内。"""
        now = time.monotonic()
        if now - self.last_check_reward_time <= REWARD_CHECK_INTERVAL:
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
        """按 timing_scale 对路线 sleep 做小幅自适应修正。"""
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
        """路线专用等待：保持时间精度，同时轮询拾取、切人、交互监听和局内检测。"""
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
        """等待一个很短的轮询间隔，用在持续检测循环里。"""
        self.sleep(0.05)
        return True

    def send_key(
        self, key, down_time=0.02, interval=-1, after_sleep=0, action_name=None
    ):
        """发送短按或长按按键，并支持动作节流和按后等待。"""
        key = _norm_key(key)
        name = action_name or f"key:{key}"
        if not self._check_interval(name, interval):
            return False
        if key == "f":
            self.ah.click_key(key, duration=DIRECT_QUICK_PICK_TAP_DURATION)
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
            tap_duration = max(float(down_time or 0.0), DIRECT_KEY_TAP_DURATION)
            if key in DIRECT_ACTION_KEYS:
                tap_duration = max(tap_duration, DIRECT_ACTION_KEY_MIN_TAP_DURATION)
            self.ah.click_key(key, duration=tap_duration)
        if after_sleep:
            self.sleep(after_sleep)
        return True

    def send_key_down(self, key, after_sleep=0):
        """按下按键并记录内部状态；F 键会转为自动连点拾取模式。"""
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
        """抬起按键并清理内部状态；F 键会停止自动连点拾取。"""
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
        """在指定时间内按固定间隔重复短按某个键。"""
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            self.send_key(key, interval=interval)
            self.sleep(0.01)

    def mouse_down(self, x=-1, y=-1, name=None, key="left"):
        """发送鼠标按下事件，主要用于长按攻击或鼠标键操作。"""
        self.ah.mouse_down(key=key)

    def mouse_up(self, name=None, key="left"):
        """发送鼠标抬起事件，配合 mouse_down 结束长按。"""
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
        """点击指定坐标；控制器可用时直接点击，否则走 MAA Click 节点。"""
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
        """通用轮询等待函数，可在每轮检测前后插入动作并要求稳定命中。"""
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
        """等待加载、黑屏或楼层切换结束，直到队伍 UI 重新稳定出现。"""
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
        """用画面亮度判断是否处于黑屏/加载状态，避免误判角色死亡。"""
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
        """检测底部队伍 UI 特征，判断当前是否已回到可操作界面。"""
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
        """截图并判断当前是否处于队伍可操作界面。"""
        image = self._screencap()
        if image is None:
            return True
        return self._is_in_team_in_image(image)

    def _current_char_roi_score(self, image, roi, index):
        """计算指定角色槽位高亮区域中的亮色/彩色像素分数。"""
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
        """计算四个角色槽位的大区域高亮分数，用于判断当前角色。"""
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
        """计算四个角色槽位的小核心高亮分数，给二号位暗头像兜底。"""
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
        """用最低分和领先差值判断目标槽位高亮是否可信。"""
        if not scores or not 0 <= index < len(scores):
            return False
        target_score = scores[index]
        other_scores = [score for idx, score in enumerate(scores) if idx != index]
        best_other = max(other_scores) if other_scores else 0
        min_score = CURRENT_CHAR_SLOT_MIN_SCORE[index]
        min_margin = CURRENT_CHAR_SLOT_MIN_MARGIN[index]
        return target_score >= min_score and target_score - best_other >= min_margin

    def _is_slot2_core_score_accepted(self, image):
        """二号位头像偏暗时，用核心高亮区域单独确认是否切到二号位。"""
        scores = self._current_char_core_scores(image)
        target_score = scores[1]
        best_other = max(score for idx, score in enumerate(scores) if idx != 1)
        return (
            target_score >= CURRENT_CHAR_SLOT2_CORE_MIN_SCORE
            and target_score - best_other >= CURRENT_CHAR_SLOT2_CORE_MIN_MARGIN
        )

    def get_current_char_index(self, image=None):
        """返回当前高亮的角色槽位索引；无法可靠判断时返回 -1。"""
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
        """判断当前高亮角色是否为指定槽位，二号位会额外走核心兜底。"""
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
        """尝试按 Esc 关闭弹窗或复活界面，直到回到队伍 UI。"""
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            if self.is_in_team():
                return True
            self.send_key("esc", action_name="ensure_in_team", interval=0.3)
            self.sleep(0.05, check_reward=False, scaled=False)
        return self.is_in_team()

    def _run_check_node(self, node_name, timeout=1.5):
        """在超时时间内反复执行某个单次识别节点，命中即返回。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            if self._recognize_once(node_name):
                return True
            time.sleep(0.01)
        return False

    def _screencap(self):
        """通过控制器截取当前游戏画面。"""
        controller = getattr(getattr(self.ctx, "tasker", None), "controller", None)
        if controller is None:
            return None
        return controller.post_screencap().wait().get()

    def _recognize_once(self, node_name, image=None):
        """执行一次识别：优先使用本地快速识别，不支持时调用 MAA 节点。"""
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

    def _find_interac_in_image(self, image, include_text=False):
        """在截图中查找可交互提示；可选文字节点兜底。"""
        if self._recognize_once("PinkPawHeist_Core3_CheckInteractPinkOnce", image):
            return True
        if self._recognize_once("PinkPawHeist_Core3_CheckInteractTemplateOnce", image):
            return True
        if not include_text:
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

    def find_interac(self, include_text=False):
        """截图后查找交互提示，用于门、锁、撤离点等交互前定位。"""
        self._checking_interaction = True
        try:
            image = self._screencap()
            return self._find_interac_in_image(image, include_text=include_text)
        finally:
            self._checking_interaction = False

    def start_interaction_watch(self):
        """开启移动过程中的交互监听，用来记录途中是否经过可交互点。"""
        self._interaction_watch_active = True
        self._interaction_watch_found = False
        return True

    def stop_interaction_watch(self):
        """关闭移动过程中的交互监听并清理监听状态。"""
        self._interaction_watch_active = False
        self._interaction_watch_found = False
        return True

    def is_lock_pick_active_fast(self):
        """截图并通过撬锁中文节点判断是否正在撬锁。"""
        image = self._screencap()
        return self._is_lock_pick_active_fast_in_image(image)

    def _is_lock_pick_active_fast_in_image(self, image):
        """在指定截图上检测“撬锁中/奋力撬锁中”状态。"""
        return self._recognize_once("PinkPawHeist_Core3_CheckLockPickActiveOnce", image)

    def is_lock_pick_active(self):
        """检测当前是否处于撬锁状态。"""
        return self.is_lock_pick_active_fast()

    def wait_lock_pick_active(self, time_out=2, settle_time=-1):
        """等待撬锁状态出现，用于确认按 F 后确实开始开锁。"""
        if self.wait_until(
            self.is_lock_pick_active_fast,
            time_out=time_out,
            settle_time=settle_time,
        ):
            return True
        return self.is_lock_pick_active()

    def is_safe_lock_pick_active(self):
        """检测保险柜撬锁状态；OCR 不稳时再用模板兜底。"""
        image = self._screencap()
        if self._is_lock_pick_active_fast_in_image(image):
            return True
        return self._recognize_once(
            "PinkPawHeist_Core3_CheckLockPickActiveTemplateOnce", image
        )

    def wait_safe_lock_pick_active(self, time_out=2, settle_time=-1):
        """等待保险柜撬锁状态出现。"""
        return self.wait_until(
            self.is_safe_lock_pick_active,
            time_out=time_out,
            settle_time=settle_time,
        )

    def wait_door_open(self, time_out=1.5):
        """用 PinkPawHeist_CheckDoorOnce 节点等待“开门”提示出现。"""
        return self._run_check_node("PinkPawHeist_CheckDoorOnce", timeout=time_out)

    def has_safe_lock_prompt(self):
        """检测保险柜旁的撬锁交互提示是否出现。"""
        image = self._screencap()
        return self._recognize_once("PinkPawHeist_Core3_CheckSafeLockPromptOnce", image)

    def wait_for_interac(self, time_out=10, include_text_fallback=True):
        """等待交互点出现；快速识别失败时可追加文字节点兜底。"""
        if self.wait_until(self.find_interac, time_out=time_out):
            return True
        if include_text_fallback:
            return self.find_interac(include_text=True)
        return False

    def wait_and_interact(
        self,
        direction=None,
        interact=True,
        key_up_sleep=None,
        is_lock=False,
        time_out=10,
    ):
        """等待交互点、按 F，并在锁类交互里确认撬锁开始/结束和保底等待。"""
        timeout = 10.0 if not time_out or time_out <= 0 else float(time_out)
        lock_min_done_at = 0.0
        direction_keys = _normalize_key_sequence(direction)

        def start_lock_min_timer():
            """锁类交互首次按 F 时启动最短等待计时，避免识别过快导致提前离开。"""
            nonlocal lock_min_done_at
            if is_lock and lock_min_done_at <= 0:
                lock_min_done_at = time.monotonic() + timeout

        def remaining_min_time():
            """计算锁类交互距离保底 time_out 还差多久。"""
            if not is_lock or lock_min_done_at <= 0:
                return 0.0
            return max(0.0, lock_min_done_at - time.monotonic())

        def wait_until_min_time():
            """锁类交互未满保底时间时补足等待。"""
            remaining = remaining_min_time()
            if remaining > 0:
                self.sleep(remaining, check_reward=False, scaled=False)

        def press_interact():
            """等待交互消失期间持续按 F，并负责启动锁类保底计时。"""
            start_lock_min_timer()
            self.send_key("f", interval=0.5)

        ret = self.wait_for_interac(time_out=timeout)
        if interact and direction_keys:
            for key in direction_keys:
                self.send_key_up(key)
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
            if not interaction_closed and self.find_interac(include_text=True):
                self.log_warning("锁交互提示仍可见，继续路线")
            return True
        wait_until_min_time()
        return True

    def loot_safes_while_walking(
        self, direction=None, min_walk_time=0, time_out=10, hold=False, send_pick=False
    ):
        """边走边处理保险柜：到时间后连点 F，遇到保险柜撬锁则停下等完成。"""
        start_time = time.monotonic()
        deadline = start_time + time_out
        earliest_lock_pick_time = start_time + min_walk_time
        if direction is not None:
            self.send_key_down(direction)
        pick_started = False

        def wait_until_pick_time():
            """行走拾取时等到最早撬锁时间，再按住 F 开始连点拾取。"""
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
        """等待保险柜撬锁开始并结束；需要时超时抛错。"""
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
        """检测撤离确认面板是否已经打开。"""
        return self._recognize_once("PinkPawHeist_CheckEvacuateOnce")

    def should_early_extract(self, exit_index):
        """读取配置，判断某个撤离点是否启用“开着就提前撤离”。"""
        if exit_index is None:
            return False
        return bool(self.early_extract_exit.get(int(exit_index), False))

    def try_open_exit(self, direction=None, exit_index=None):
        """尝试与撤离点交互；确认可撤离后记录出口状态或直接提前撤离。"""
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
        """一边朝撤离点走一边按 F，直到撤离确认面板出现。"""
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
        """切到战斗角色清怪，确认无怪后切回跑图角色。"""
        self.switch_to_fighter(check_switched=True, mode=fighter_mode)
        self.fight_until_no_monster(timeout_no_monster=10000, wait_for_monster=True)
        self.switch_to_runner(check_switched=True)

    def check_monster(self):
        """通过血条颜色节点判断当前画面是否有怪物。"""
        image = self.ctx.tasker.controller.post_screencap().wait().get()
        result = self.ctx.run_recognition("PinkPawHeist_CheckMonsterOnce", image)
        return result is not None and getattr(result, "hit", False)

    def wait_monster(self, timeout=6000):
        """在指定时间内等待怪物出现。"""
        deadline = time.monotonic() + timeout / 1000.0
        while time.monotonic() < deadline:
            if self.check_monster():
                return True
            self.sleep(0.2)
        return False

    def attack_cycle(self, times=3, loot=False):
        """执行一轮基础攻击动作，必要时顺手按 F 拾取。"""
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
        """循环攻击直到持续一段时间检测不到怪物。"""
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
        """直接多次短按指定角色键，适合不需要确认的强制切人。"""
        for _ in range(4):
            self.send_key(str(key))
            self.sleep(0.2)
        return str(key)

    def _send_current_switch_key(self):
        """发送当前候选角色键，并刷新切换确认截止时间。"""
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
        """清空正在进行的角色切换状态。"""
        self._switch_state = None
        self._next_switch_poll_at = 0.0

    def _handle_dead_switch_candidate(self, state: CharacterSwitchState):
        """切人后疑似不在队伍 UI 时，按角色死亡处理并尝试下一个候选。"""
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
        """后台监控未确认切人过程，处理黑屏、死亡和复活界面。"""
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
        """等待目标槽位高亮确认；没确认时重按，疑似死亡时换下一个候选。"""
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
        """创建切人状态并发出首个候选按键，必要时等待高亮确认。"""
        keys = [str(key) for key in keys]
        if not keys:
            raise AbortException(f"{role} {keys} dead or empty")
        self._switch_state = CharacterSwitchState(role=role, keys=keys)
        key = self._send_current_switch_key()
        if check_switched:
            return self._wait_character_switch_success(role, key)
        return key

    def switch_to_runner(self, check_switched=False):
        """切到跑图角色，默认是三号位薄荷。"""
        return self._begin_character_switch(
            self.ROLE_RUNNER, self.config.get(self.CONF_RUNNER, []), check_switched
        )

    def switch_to_avoider(self, check_switched=False):
        """切到避战角色；未配置避战角色时直接返回。"""
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            self.log_info("no avoider")
            return None
        return self._begin_character_switch(self.ROLE_AVOIDER, keys, check_switched)

    def avoider_strategy_index(self):
        """根据避战方式返回路线分支：无避战、早雾/翳、浔。"""
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            return -1
        method_name = self.config.get(self.CONF_AVOID_MTH)
        if method_name not in self.avoid_methods:
            return 0
        return self.avoid_methods.index(method_name)

    def perform_avoidance_action(self):
        """执行当前避战动作：翳长按 Shift 或浔长按攻击。"""
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
        """点击确认撤离，等待收益识别并记录奖励。"""
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
        """路线异常时释放按键、退出界面并记录失败。"""
        self.log_round_info("Abort and return to main")
        self.ah.release_controls()
        for _ in range(4):
            self.send_key("esc")
            self.sleep(1.0, check_reward=False, scaled=False)
        self.ah.run_task("PinkPawHeist_Once")
        self.sleep(5.0, check_reward=False, scaled=False)
        notify_pinkpaw_reward(self.ctx, success=False)

    def _release_held_keys(self):
        """释放脚本内部记录为按住状态的键，防止异常后持续移动。"""
        held = list(self._held_keys)
        self._held_keys.clear()
        for key in held:
            try:
                self.ah.key_up(key)
            except Exception as exc:
                self.log_error(f"release held key {key} failed", exc)
        self._quick_pick_active = False

    def goto_lg1(self):
        """原始开局路线：从小吱大厅进入 LG1，并处理开锁、保险柜和清怪。"""
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
        start = time.monotonic()
        while time.monotonic() < start + 10:
            self.send_key("space", down_time=0.14, interval=0.25)
            if time.monotonic() > start + 6.4 and self.find_interac():
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
        """LG1 开锁被怪打断后的恢复路线，清怪后回到门前继续。"""
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
        """LG1 第一段搜刮路线：经过保险柜区并保持拾取。"""
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
        """LG1 第二段路线：穿过镭射区域并继续搜刮。"""
        self.log_round_info("LG1 WP2")
        self.send_key_down("d")
        self.sleep(1.80)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.68)
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
        self.sleep(1.46)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(2.78)
        self.send_key_down("w")  # 过镭射1
        self.sleep(1.81)
        self.send_key_up("w")
        self.sleep(0.54)
        self.start_interaction_watch()
        self.send_key_down("w")  # 过镭射2
        self.sleep(8.51)
        self.send_key_up("w")
        self.stop_interaction_watch()
        self.sleep(0.33)

    def lg1_wp3(self):
        """LG1 第三段路线：继续保险柜搜刮并穿过后续走廊。"""
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
        self.sleep(3.23)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(1.86)
        self.send_key_up("w")
        self.send_key_down("d")
        self.sleep(1.41)
        self.send_key_up("d")
        self.send_key_down("w")
        self.sleep(2.33)
        self.send_key_up("f")  # end pick
        self.send_key_down("lshift")
        self.sleep(0.10)
        self.send_key_up("lshift")
        self.sleep(0.23)
        self.send_key_up("w")
        self.sleep(0.53)
        self.send_key_down("w")
        self.sleep(0.12)
        self.send_key_down("a")
        self.sleep(0.13)
        self.send_key_up("a")
        self.sleep(4.03)
        self.send_key_up("w")
        self.sleep(0.12)

    def lg1_wp4(self):
        """LG1 第四段通用路线：处理跳跃、拾取和长距离移动。"""
        self.log_round_info("LG1 WP4")
        self.send_key_down("d")
        self.sleep(0.21)
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
        """LG1 第五段原始避战路线：无专用避战角色时使用。"""
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

        deadline = time.monotonic() + 4.5
        while time.monotonic() < deadline:
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
        """LG1 第五段早雾/翳避战路线：先避战再进门。"""
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
        """LG1 第五段浔避战路线：用长按攻击规避战斗。"""
        self.log_round_info("LG1 WP5避战路线3")
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.5)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.send_key_down("w")
        self.sleep(0.11)

        deadline = time.monotonic() + 4.5
        while time.monotonic() < deadline:
            self.send_key("lshift")
            self.sleep(0.51)

        self.send_key_up("w")
        self.sleep(0.2)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.5)
        self.send_key_down("d")
        self.sleep(0.30)
        self.send_key_up("d")
        self.wait_and_interact(is_lock=True)
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.25)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg2_wp1_to_exit1(self):
        """LG2 第一段路线：搜刮后尝试第一个撤离点是否可用。"""
        self.log_round_info("LG2 WP1尝试出口1")
        self.sleep(2.65)  # 2.65
        self.send_key_down("w")
        self.sleep(4.95)
        self.send_key_up("w")
        self.sleep(0.13)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(2.70)
        self.send_key("lshift")  # x0.6
        self.sleep(3.30)
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
        self.sleep(2.53)
        self.send_key_down("a")
        self.sleep(0.15)
        self.send_key_up("a")
        self.exit_state[1] = self.try_open_exit(direction="w", exit_index=1)

    def lg2_wp1_remains(self):
        """第一个撤离点不可直接撤时，继续执行 LG2 WP1 剩余搜刮路线。"""
        self.log_round_info("LG2 WP1剩余路线")
        self.send_key_down("w")
        self.sleep(2.14)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep(0.90)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("w")
        self.sleep(0.80)
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
        self.sleep(0.81)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.2)
        self.send_key("lshift")
        self.sleep(1.40)
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
        """LG2 第二段路线：搜刮后尝试第二个撤离点是否可用。"""
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
        """从 LG2 WP3 前往镭射房入口的路线。"""
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
        """镭射房内部搜刮路线，包含多次跳跃、保险柜和拾取。"""
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
        self.sleep(0.15)
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
        self.sleep(0.11)
        self.send_key_down("space")
        self.sleep(0.17)
        self.send_key_up("space")
        self.sleep(1.18)
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
        self.sleep(0.64)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.53)
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
        self.sleep(0.35)
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
        self.sleep(0.58)
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
        """LG2 后段公共路线，进入最终撤离路线选择前的位置。"""
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
        """出口1可用时，从 LG2 WP4 前往出口1撤离。"""
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

        deadline = time.monotonic() + 1.29
        while time.monotonic() < deadline:
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
        """出口2可用时，从 LG2 WP4 前往出口2撤离。"""
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
        """出口1/2都不可用时，前往默认出口3撤离。"""
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
        """按所选配队/避战分支执行完整 Core3 路线，并根据出口状态选择撤离路线。"""
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
            self.lg1_wp4_buster()
            self.lg1_wp5_buster2()
        self.wait_team_ui_settle()
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
        """早雾分支开局：用翳/早雾处理大厅避战和控怪后进入 LG1。"""
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
        self.send_key("a", down_time=0.23)
        self.sleep(0.10)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.4)
        if self.find_interac():
            self.send_key("s", down_time=0.10)
            self.sleep(0.25)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key("e", down_time=2.40)
            self.send_key_down("w")
            self.wait_and_interact(direction="w", is_lock=True, time_out=5.4)
        self.send_key("s", down_time=0.10)
        self.switch_to_avoider(check_switched=True)  # 切到狗哥潜行避免碰到怪改变路径
        self.send_key_down("w")
        self.sleep(0.35)
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
        self.wait_and_interact(direction="wd", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_up("w")
        if self.wait_door_open(time_out=1.14):
            self.sleep(0.10)
            self.send_key("f", down_time=0.10)
            self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.40)
        self.send_key("a", down_time=0.36)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def goto_lg1_skip_Hotori(self):
        """浔分支开局：用浔长按攻击规避大厅战斗后进入 LG1。"""
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
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.60)
        self.switch_to_fighter(
            check_switched=True, mode=1
        )  # 切到狗哥潜行避免碰到怪改变路径
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
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.10)
        self.send_key("a", down_time=0.22)
        self.sleep(0.10)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.4)
        self.sleep(0.01)
        self.send_key_down("w")
        self.sleep(0.38)
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
        self.sleep(0.50)
        self.send_key("lshift", down_time=0.20)
        self.sleep(2.30)
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
        self.send_key("s", down_time=1.26)
        self.sleep(0.10)
        self.send_key("d", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=1.88)
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
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.42)
        self.send_key_down("d")
        self.sleep(0.43)
        self.send_key_up("d")
        self.sleep(0.70)
        self.send_key_up("w")
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(1.50)
        self.send_key_down("d")
        self.sleep(0.10)
        self.wait_and_interact(direction="wd", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_up("w")
        if self.wait_door_open(time_out=1.14):
            self.sleep(0.10)
            self.send_key("f", down_time=0.10)
            self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("a", down_time=0.36)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def lobby_open_door_check(self, check_time=3):
        """大厅门口循环检测“开门”节点，没开就按 F 再试。"""
        open_door = False
        open_loop = 0
        while not open_door and open_loop < check_time:
            if self.wait_door_open(time_out=1.14):
                open_door = True
            else:
                self.sleep(0.10)
                self.send_key("f", down_time=0.10)
                self.sleep(0.20)
                open_loop += 1
        return open_door

    def lg1_wp1_safer(self):
        """更稳的 LG1 WP1 路线：少停门口，优先保证安全通过。"""
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
        self.sleep(1.22)
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
        self.sleep(1.98)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.21)
        self.send_key_up("d")
        self.sleep(0.12)

    def lg1_wp4_buster(self):
        """早雾/翳分支使用的 LG1 WP4 变体路线。"""
        self.log_round_info("LG1 WP4 bUSTER")
        self.send_key_down("d")
        self.sleep(0.21)
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
        """早雾/翳分支使用的 LG1 WP5 避战与进门路线。"""
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
        self.sleep(0.30)
        self.send_key_up("d")
        self.wait_and_interact(is_lock=True)
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(0.25)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.10)
        self.wait_and_interact(direction="w")

    def lg1_wp5_buster2(self):
        """浔/翳分支使用的 LG1 WP5 避战与进门路线。"""
        self.log_round_info("LG1 WP5 Buster 开始避战路线")
        self.switch_to_fighter(check_switched=True, mode=1)
        self.sleep(0.50)
        self.send_key_down("w")
        self.sleep(0.1)
        self.send_key_down("lshift")
        self.sleep(1.0)
        self.send_key_up("lshift")
        self.sleep(0.1)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(6.00)
        self.send_key_up("w")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.32)
        self.send_key_down("d")
        self.sleep(0.30)
        self.send_key_up("d")
        self.wait_and_interact(is_lock=True)
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(0.25)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.10)
        self.wait_and_interact(direction="w")

    def lg2_wp2_to_exit2_safer(self):
        """更稳的 LG2 WP2 路线：调整前半段走位后再尝试第二撤离点。"""
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
        self.sleep(0.70)
        self.send_key_up("a")
        self.sleep(1.26)
        self.send_key_down("w")
        self.sleep(2.62)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.13)
        self.send_key("lshift", down_time=0.10)
        self.sleep(0.21)
        self.send_key("lshift", down_time=0.10)
        self.sleep(1.01)
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

    def switch_to_fighter(self, check_switched=False, mode="all_desc"):
        """切换到战斗角色；可按升序、降序或指定排序位置选择候选。"""
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
        """MAA 自定义动作入口：创建 Core3 路线对象，执行路线，并统一处理停止、提前撤离和异常恢复。"""
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
            input_mode = "direct" if path.ah.direct_input.available else "maa"
            path.log_round_info(
                f"Start, method {path.config[path.CONF_AVOID_MTH]}, timing x{path.route_timing_scale:.2f}, input {input_mode}"
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
