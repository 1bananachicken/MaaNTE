import ctypes
import re
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

EnumWindows = user32.EnumWindows
EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
EnumWindows.restype = wintypes.BOOL

GetWindowTextLengthW = user32.GetWindowTextLengthW
GetWindowTextLengthW.argtypes = [wintypes.HWND]
GetWindowTextLengthW.restype = ctypes.c_int

GetWindowTextW = user32.GetWindowTextW
GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GetWindowTextW.restype = ctypes.c_int

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = wintypes.HWND

SetForegroundWindow = user32.SetForegroundWindow
SetForegroundWindow.argtypes = [wintypes.HWND]
SetForegroundWindow.restype = wintypes.BOOL

IsIconic = user32.IsIconic
IsIconic.argtypes = [wintypes.HWND]
IsIconic.restype = wintypes.BOOL

ShowWindow = user32.ShowWindow
ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
ShowWindow.restype = wintypes.BOOL

GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
GetWindowThreadProcessId.restype = wintypes.DWORD

AttachThreadInput = user32.AttachThreadInput
AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
AttachThreadInput.restype = wintypes.BOOL

GetCurrentThreadId = kernel32.GetCurrentThreadId
GetCurrentThreadId.restype = wintypes.DWORD

SW_RESTORE = 9
SW_SHOW = 5

_cached_hwnd = None
_cached_title = None


def _search_hwnd(window_title_regex):
    results = []

    def callback(hwnd, _lparam):
        if not IsWindowVisible(hwnd):
            return True
        length = GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        if buf.value and re.search(window_title_regex, buf.value):
            results.append(hwnd)
            return False
        return True

    EnumWindows(WNDENUMPROC(callback), 0)
    return results[0] if results else None


def find_game_window(window_title_regex="异环"):
    global _cached_hwnd, _cached_title

    if _cached_hwnd is not None and _cached_title == window_title_regex:
        return _cached_hwnd

    hwnd = _search_hwnd(window_title_regex)
    if hwnd is not None:
        _cached_hwnd = hwnd
        _cached_title = window_title_regex
    return hwnd


def is_game_window_foreground(window_title="异环"):
    hwnd = find_game_window(window_title)
    if hwnd is None:
        return False
    fg_hwnd = GetForegroundWindow()
    return hwnd == fg_hwnd


def ensure_game_window_foreground(window_title="异环"):
    hwnd = find_game_window(window_title)
    if hwnd is None:
        raise RuntimeError(f"未找到游戏窗口: {window_title}")

    if hwnd == GetForegroundWindow():
        return True

    if IsIconic(hwnd):
        ShowWindow(hwnd, SW_RESTORE)
    else:
        ShowWindow(hwnd, SW_SHOW)

    fg_thread = GetWindowThreadProcessId(GetForegroundWindow(), None)
    current_thread = GetCurrentThreadId()

    if fg_thread != current_thread and fg_thread != 0:
        AttachThreadInput(current_thread, fg_thread, True)

    result = SetForegroundWindow(hwnd)

    if fg_thread != current_thread and fg_thread != 0:
        AttachThreadInput(current_thread, fg_thread, False)

    if not result:
        raise RuntimeError(f"无法将游戏窗口置顶: {window_title}")

    return True
