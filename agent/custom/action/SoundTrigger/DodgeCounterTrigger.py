import ctypes
import random
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

VK_SHIFT = 0xA0

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MAPVK_VK_TO_VSC = 0

user32 = ctypes.WinDLL("user32", use_last_error=True)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    )

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        if not self.dwFlags & KEYEVENTF_KEYUP:
            self.wScan = user32.MapVirtualKeyExW(self.wVk, MAPVK_VK_TO_VSC, 0)


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD), ("_input", _INPUT))


def _send_key(key, up=False):
    flags = KEYEVENTF_KEYUP if up else 0
    x = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=key, dwFlags=flags))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def _send_mouse(flags=0):
    x = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, flags, 0, 0))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


logger = None


def _log():
    global logger
    if logger is None:
        from custom.action.Common.logger import get_logger
        logger = get_logger(__name__)
    return logger


class Dodger:
    def __init__(
        self,
        dodge_fn: Optional[Callable] = None,
        counter_fn: Optional[Callable] = None,
        stop_check=None,
    ):
        self.dodge_fn = dodge_fn or self._default_dodge
        self.counter_fn = counter_fn or self._default_counter
        self.stop_check = stop_check

        self._busy = False
        self._lock = threading.Lock()
        self._last_dodge = 0.0
        self._last_counter = 0.0
        self._dodge_cd = 0.5
        self._counter_cd = 1.0

    def dodge(self):
        if self.stop_check and self.stop_check():
            return
        if time.time() - self._last_dodge < self._dodge_cd:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.dodge_fn()
            self._last_dodge = time.time()
        except Exception as e:
            _log().error(f"Dodge failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def counter(self):
        if self.stop_check and self.stop_check():
            return
        if time.time() - self._last_counter < self._counter_cd:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.counter_fn()
            self._last_counter = time.time()
        except Exception as e:
            _log().error(f"Counter failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _default_dodge(self):
        _log().info("执行按键: 右键按下 -> 右键松开 -> 左Shift按下 -> 左Shift松开")
        _send_mouse(MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.1 + random.random() * 0.2)
        _send_mouse(MOUSEEVENTF_RIGHTUP)
        time.sleep(0.1)
        _send_key(VK_SHIFT)
        time.sleep(0.1 + random.random() * 0.1)
        _send_key(VK_SHIFT, up=True)

    def _default_counter(self):
        key = random.choice([0x31, 0x32, 0x33, 0x34])
        key_name = chr(key)
        _log().info(f"执行按键: {key_name}按下 -> 左键按下 -> 左键松开 -> {key_name}松开")
        _send_key(key)
        time.sleep(0.1)
        _send_mouse(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.05)
        _send_mouse(MOUSEEVENTF_LEFTUP)
        _send_key(key, up=True)
