import ctypes
import os
import re
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.SetWindowLongW.restype = wintypes.LONG
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HANDLE
user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
user32.GetMonitorInfoW.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.IsZoomed.argtypes = [wintypes.HWND]
user32.IsZoomed.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindowEnabled.argtypes = [wintypes.HWND]
user32.IsWindowEnabled.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
kernel32.Sleep.argtypes = [wintypes.DWORD]

# 使进程感知 DPI，避免 GetClientRect 返回缩放后的虚拟坐标
# 150% 缩放时未设置此项会导致返回值只有实际分辨率的 2/3
user32.SetProcessDPIAware()

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL

TH32CS_SNAPPROCESS = 0x00000002
DEFAULT_GAME_PROCESS_NAME = "HTGame.exe"
DEFAULT_WINDOW_RESIZE_SETTLE_MS = 300
SW_RESTORE = 9
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_POPUP = 0x80000000
SM_CXSCREEN = 0
SM_CYSCREEN = 1
SWP_NOSIZE = 0x0001
SWP_NOREPOSITION = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOMOVE = 0x0002
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
MONITOR_DEFAULTTONEAREST = 0x00000002


def _log(message):
    print(f"[Win32Process] {message}")


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def _normalize_process_names(process_name):
    if isinstance(process_name, str):
        items = [process_name]
    else:
        items = list(process_name or [])
    names = []
    for item in items:
        name = os.path.basename(str(item)).strip().lower()
        if name and name not in names:
            names.append(name)
    return names


def get_pids_by_name(process_name):
    process_names = set(_normalize_process_names(process_name))
    if not process_names:
        return []
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    pids = []
    if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
        while True:
            if entry.szExeFile.lower() in process_names:
                pids.append(entry.th32ProcessID)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    kernel32.CloseHandle(snapshot)
    return pids


def get_window_text(hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def get_class_name(hwnd):
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _match_class_name(class_name, patterns):
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if isinstance(pattern, str):
            if class_name == pattern:
                return True
        elif re.search(pattern, class_name):
            return True
    return False


def find_windows_by_process(process_name, hwnd_class=None, require_title=False):
    pids = get_pids_by_name(process_name)
    if not pids:
        return []
    pid_set = set(pids)
    results = []

    def callback(hwnd, _lparam):
        if not user32.IsWindow(hwnd) or not user32.IsWindowEnabled(hwnd):
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in pid_set:
            return True
        class_name = get_class_name(hwnd)
        if not _match_class_name(class_name, hwnd_class):
            return True
        title = get_window_text(hwnd)
        if require_title and not title:
            return True
        client_size = get_client_size(hwnd)
        if client_size is None or client_size[0] <= 10 or client_size[1] <= 10:
            return True
        window_rect = get_window_rect(hwnd)
        if window_rect is None:
            return True
        results.append(
            {
                "hwnd": hwnd,
                "client_size": client_size,
                "client_area": client_size[0] * client_size[1],
                "window_rect": window_rect,
                "title": title,
                "class_name": class_name,
                "order": len(results),
            }
        )
        return True

    callback_ref = WNDENUMPROC(callback)
    user32.EnumWindows(callback_ref, 0)
    return results


def find_window_by_process(
    process_name,
    hwnd_class=None,
    require_title=False,
    selected_hwnd=0,
    last_hwnd=0,
):
    windows = find_windows_by_process(
        process_name,
        hwnd_class=hwnd_class,
        require_title=require_title,
    )
    if not windows:
        return None

    selected = next(
        (item for item in windows if selected_hwnd and item["hwnd"] == selected_hwnd),
        None,
    )
    biggest = max(windows, key=lambda item: item["client_area"])
    if selected is not None:
        return selected["hwnd"]

    last = next(
        (item for item in windows if last_hwnd and item["hwnd"] == last_hwnd),
        None,
    )
    if last is not None and biggest["client_area"] <= last["client_area"] * 1.1:
        return last["hwnd"]
    return biggest["hwnd"]


def get_client_size(hwnd):
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.right - rect.left, rect.bottom - rect.top


def get_window_rect(hwnd):
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def _get_monitor_work_area(hwnd):
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    rect = info.rcWork
    return rect.left, rect.top, rect.right, rect.bottom


def show_title_bar(hwnd):
    """make sure the target window has a normal title bar."""
    try:
        current_style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        if current_style & WS_CAPTION:
            return True
        new_style = (int(current_style) | WS_CAPTION) & ~WS_POPUP
        user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
        user32.SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
        kernel32.Sleep(10)
        return bool(user32.GetWindowLongW(hwnd, GWL_STYLE) & WS_CAPTION)
    except Exception:
        return False


def resize_window(hwnd, width, height, center=True):
    """Resize the outer window and center it, following OK-NTE's sequence."""
    if not hwnd:
        return False
    width = int(width)
    height = int(height)
    flags = SWP_SHOWWINDOW | SWP_NOZORDER | SWP_NOREPOSITION
    if not user32.SetWindowPos(hwnd, None, 0, 0, width, height, flags):
        return False
    kernel32.Sleep(10)

    expected_left = None
    expected_top = None
    if center:
        rect = get_window_rect(hwnd)
        if rect is None:
            return False
        left, top, right, bottom = rect
        window_width = right - left
        window_height = bottom - top
        screen_width = user32.GetSystemMetrics(SM_CXSCREEN)
        screen_height = user32.GetSystemMetrics(SM_CYSCREEN)
        expected_left = (screen_width - window_width) // 2
        expected_top = (screen_height - window_height) // 2
        if not user32.SetWindowPos(
            hwnd,
            None,
            expected_left,
            expected_top,
            0,
            0,
            SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW,
        ):
            return False

    for _ in range(50):
        rect = get_window_rect(hwnd)
        if rect is None:
            return False
        left, top, right, bottom = rect
        current_width = right - left
        current_height = bottom - top
        size_ok = current_width == width and current_height == height
        pos_ok = not center or (left == expected_left and top == expected_top)
        if size_ok and pos_ok:
            break
        kernel32.Sleep(100)
    kernel32.Sleep(500)
    return True


def resize_client_area(hwnd, width, height, center=True, tolerance=2):
    """Resize a window so its client area matches the target size."""
    if not hwnd:
        return False
    target_width = int(width)
    target_height = int(height)
    if user32.IsIconic(hwnd) or user32.IsZoomed(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        kernel32.Sleep(100)
    show_title_bar(hwnd)

    current_client = get_client_size(hwnd)
    current_rect = get_window_rect(hwnd)
    if current_client is None or current_rect is None:
        return False
    if (
        abs(current_client[0] - target_width) <= tolerance
        and abs(current_client[1] - target_height) <= tolerance
    ):
        return True

    left, top, right, bottom = current_rect
    window_width = right - left
    window_height = bottom - top
    border = max(0, window_width - current_client[0])
    title_height = max(0, window_height - current_client[1])
    resized_width = target_width + border
    resized_height = target_height + title_height
    screen_width = user32.GetSystemMetrics(SM_CXSCREEN)
    screen_height = user32.GetSystemMetrics(SM_CYSCREEN)
    if screen_width < resized_width or screen_height < resized_height:
        return False

    if not resize_window(hwnd, resized_width, resized_height, center=center):
        return False

    for _ in range(20):
        current_client = get_client_size(hwnd)
        if current_client is not None and (
            abs(current_client[0] - target_width) <= tolerance
            and abs(current_client[1] - target_height) <= tolerance
        ):
            return True
        kernel32.Sleep(50)
    return False


def ensure_process_client_size(
    process_name,
    width,
    height,
    center=True,
    tolerance=2,
    settle_ms=300,
):
    """Find a process window and resize its client area to the target size."""
    hwnd = find_window_by_process(process_name)
    if not hwnd:
        return {
            "success": False,
            "reason": "window_not_found",
            "hwnd": None,
            "before": None,
            "after": None,
        }

    target = (int(width), int(height))
    before = get_client_size(hwnd)
    if before is None:
        return {
            "success": False,
            "reason": "client_size_unavailable",
            "hwnd": hwnd,
            "before": None,
            "after": None,
        }

    if (
        abs(before[0] - target[0]) <= tolerance
        and abs(before[1] - target[1]) <= tolerance
    ):
        return {
            "success": True,
            "reason": "already_matched",
            "hwnd": hwnd,
            "before": before,
            "after": before,
        }

    resized = resize_client_area(
        hwnd,
        target[0],
        target[1],
        center=center,
        tolerance=tolerance,
    )
    if resized and settle_ms:
        kernel32.Sleep(int(settle_ms))
    after = get_client_size(hwnd)
    if resized:
        _log(
            f"game window resolution {before[0]}x{before[1]} -> {target[0]}x{target[1]}"
        )
    return {
        "success": bool(resized),
        "reason": "resized" if resized else "resize_failed",
        "hwnd": hwnd,
        "before": before,
        "after": after,
    }


def ensure_game_window_resolution(
    width,
    height,
    process_name=DEFAULT_GAME_PROCESS_NAME,
    settle_ms=DEFAULT_WINDOW_RESIZE_SETTLE_MS,
    **kwargs,
):
    """Resize the game window client area to the target resolution."""
    return ensure_process_client_size(
        process_name,
        width,
        height,
        settle_ms=settle_ms,
        **kwargs,
    )
