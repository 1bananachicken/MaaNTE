import ctypes
import time

from .key_mapping import NOTE_KEY_MAPPING
from utils.logger import logger

user32 = ctypes.windll.user32
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_ACTIVATE = 0x0006
WA_CLICKACTIVE = 2

# 使用游戏专用的左 Shift 和左 Ctrl 防跑调。
WIN32_VK = {
    "shift": 0xA0,
    "ctrl": 0xA2,
    "a": 0x41,
    "b": 0x42,
    "c": 0x43,
    "d": 0x44,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "i": 0x49,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "q": 0x51,
    "r": 0x52,
    "s": 0x53,
    "t": 0x54,
    "u": 0x55,
    "v": 0x56,
    "w": 0x57,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,
}

WINDOW_TITLES = [
    "NTE  ",
    "异环  ",
]


def get_lparam(vk_code, is_down=True):
    """构建底层硬件扫描码。"""
    scan_code = user32.MapVirtualKeyW(vk_code, 0)
    lparam = 1 | (scan_code << 16)
    if not is_down:
        lparam |= 0xC0000000
    return lparam


class MaaKeyboardBridge:
    def __init__(
        self,
        hold_seconds: float = 0.008,
        mapping: dict | None = None,
    ):
        self.mapping = mapping if mapping is not None else NOTE_KEY_MAPPING
        self.hold_seconds = hold_seconds
        self.hwnd = 0

        for title in WINDOW_TITLES:
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                self.hwnd = hwnd
                logger.info("已连接到游戏窗口: '%s' (HWND: %s)", title, self.hwnd)
                break

        if not self.hwnd:
            logger.warning(
                "未找到列表中的任何窗口，请检查游戏是否运行！列表: %s",
                WINDOW_TITLES,
            )

    def _force_send_key(self, vk_code, is_down):
        """通过 PostMessage 向游戏窗口发送按键。"""
        if not self.hwnd:
            return

        lparam = get_lparam(vk_code, is_down)
        msg = WM_KEYDOWN if is_down else WM_KEYUP
        user32.PostMessageW(self.hwnd, msg, vk_code, lparam)

    def _activate(self):
        user32.SendMessageW(self.hwnd, WM_ACTIVATE, WA_CLICKACTIVE, 0)

    def execute_chord(self, midi_notes):
        """按修饰键类型隔离并短按一个和弦。"""
        if not self.hwnd:
            return

        self._activate()
        self._execute_chord_groups(midi_notes)

    def _execute_chord_groups(self, midi_notes):
        """发送已经激活窗口的短按和弦。"""

        normal_keys, shift_keys, ctrl_keys = [], [], []

        for note in midi_notes:
            if note not in self.mapping:
                continue
            action = self.mapping[note]
            key = action.split("+")[-1]
            if "shift+" in action:
                shift_keys.append(key)
            elif "ctrl+" in action:
                ctrl_keys.append(key)
            else:
                normal_keys.append(key)

        self._press_group(normal_keys)
        self._press_group(shift_keys, "shift")
        self._press_group(ctrl_keys, "ctrl")

    def _press_group(self, keys, modifier: str | None = None):
        if not keys:
            return

        if modifier and modifier in WIN32_VK:
            self._force_send_key(WIN32_VK[modifier], True)
            time.sleep(0.002)

        vk_codes = [WIN32_VK[key] for key in keys if key in WIN32_VK]
        for vk in vk_codes:
            self._force_send_key(vk, True)

        if self.hold_seconds > 0:
            time.sleep(self.hold_seconds)

        for vk in reversed(vk_codes):
            self._force_send_key(vk, False)

        if modifier and modifier in WIN32_VK:
            time.sleep(0.001)
            self._force_send_key(WIN32_VK[modifier], False)
