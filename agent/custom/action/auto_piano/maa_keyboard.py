import time
import ctypes
from .key_mapping import NOTE_KEY_MAPPING

user32 = ctypes.windll.user32
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_ACTIVATE = 0x0006
WA_CLICKACTIVE = 2

# 使用游戏专用的 左Shift 和 左Ctrl 防跑调
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

# ==========================================
# 填入游戏的精确名称
WINDOW_TITLE = "异环  "
# ==========================================


def get_lparam(vk_code, is_down=True):
    """构建底层硬件扫描码"""
    scan_code = user32.MapVirtualKeyW(vk_code, 0)
    lparam = 1 | (scan_code << 16)
    if not is_down:
        lparam |= 0xC0000000
    return lparam


class MaaKeyboardBridge:
    def __init__(
        self, controller=None, hold_seconds: float = 0.008, wait_jobs: bool = False
    ):
        self.mapping = NOTE_KEY_MAPPING
        self.hold_seconds = hold_seconds
        self.hwnd = user32.FindWindowW(None, WINDOW_TITLE)

        if not self.hwnd:
            print(f"【警告】找不到窗口: '{WINDOW_TITLE}'")

    def _force_send_key(self, vk_code, is_down):
        """带有强行唤醒的底层发送器"""
        if not self.hwnd:
            return

        # 发送前强行给窗口发一个“鼠标点击激活”信号，骗过后台检测！
        user32.SendMessageW(self.hwnd, WM_ACTIVATE, WA_CLICKACTIVE, 0)

        lparam = get_lparam(vk_code, is_down)
        msg = WM_KEYDOWN if is_down else WM_KEYUP
        user32.PostMessageW(self.hwnd, msg, vk_code, lparam)

    def execute_chord(self, midi_notes):
        if not self.hwnd:
            return

        normal_keys, shift_keys, ctrl_keys = [], [], []

        for note in midi_notes:
            if note in self.mapping:
                action = self.mapping[note]
                key = action.split("+")[-1]
                if "shift+" in action:
                    shift_keys.append(key)
                elif "ctrl+" in action:
                    ctrl_keys.append(key)
                else:
                    normal_keys.append(key)

        # 隔离分组，并发弹奏
        self._press_group(normal_keys, None)
        self._press_group(shift_keys, "shift")
        self._press_group(ctrl_keys, "ctrl")

    def _press_group(self, keys, modifier: str | None = None):
        if not keys:
            return

        # 1. 按下修饰键
        if modifier and modifier in WIN32_VK:
            self._force_send_key(WIN32_VK[modifier], True)
            time.sleep(0.002)  # 微小停顿，防跑调

        # 2. 批量按下音符键
        vk_codes = [WIN32_VK[key] for key in keys if key in WIN32_VK]
        for vk in vk_codes:
            self._force_send_key(vk, True)

        # 3. 极速停留
        if self.hold_seconds > 0:
            time.sleep(self.hold_seconds)

        # 4. 批量抬起音符键
        for vk in reversed(vk_codes):
            self._force_send_key(vk, False)

        # 5. 抬起修饰键
        if modifier and modifier in WIN32_VK:
            time.sleep(0.001)
            self._force_send_key(WIN32_VK[modifier], False)
