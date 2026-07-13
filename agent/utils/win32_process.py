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

# GeForce NOW 云游戏窗口特征
# Chrome 网页版：标题已实测确认（screenshot/NTE window name.png）
GFN_CHROME_PROCESS_NAME = "chrome.exe"
GFN_CHROME_WINDOW_CLASS = "Chrome_WidgetWin_1"
GFN_CHROME_TITLE_REGEX = r"NTE.*on GeForce NOW"
# 原生客户端：标题已实测确认（GFNWindowMover 进程选择器截图），与 Chrome 版一致；
# 窗口类已运行时实测确认为 CEFCLIENT（PRD 风险 R3 已关闭，interface.json 控制器
# 按此匹配）。agent 侧探测仍不过滤类名：进程名 + 标题正则已足够唯一，
# 且可兼容未来客户端更新导致的类名变更（探测日志保留 class= 字段供核对）
GFN_APP_PROCESS_NAME = "GeForceNOW.exe"
GFN_APP_TITLE_REGEX = r"NTE.*on GeForce NOW"

# 游戏窗口运行模式（detect_game_window 的探测结果）
GAME_WINDOW_MODE_NATIVE = "native"
GAME_WINDOW_MODE_GFN_CHROME = "gfn_chrome"
GAME_WINDOW_MODE_GFN_APP = "gfn_app"
GAME_WINDOW_MODE_NOT_FOUND = "not_found"
SW_RESTORE = 9
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_POPUP = 0x80000000
SM_CXSCREEN = 0
SM_CYSCREEN = 1
SWP_NOSIZE = 0x0001
SWP_NOREPOSITION = 0x0200
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOMOVE = 0x0002
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
MONITOR_DEFAULTTONEAREST = 0x00000002


def _log(message, level="info"):
    """经项目统一 logger 输出（文件滚动/级别控制/MXU 控制台格式），
    保留 [Win32Process] 前缀便于按模块过滤。"""
    from utils.logger import logger

    getattr(logger, level, logger.info)("[Win32Process] %s", message)


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


def _match_title(title, patterns):
    """标题正则过滤。None/空列表 表示不过滤；str 或 str 列表按 re.search 匹配。

    空列表与 None 同义（不过滤）：避免动态构建模式列表为空时
    静默过滤掉所有窗口。
    """
    if not patterns:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        if re.search(pattern, title):
            return True
    return False


def find_windows_by_process(
    process_name, hwnd_class=None, require_title=False, title_regex=None
):
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
        if not _match_title(title, title_regex):
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
    title_regex=None,
    selected_hwnd=0,
    last_hwnd=0,
):
    windows = find_windows_by_process(
        process_name,
        hwnd_class=hwnd_class,
        require_title=require_title,
        title_regex=title_regex,
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


def detect_game_window(selected_hwnd=0, last_hwnd=0):
    """按优先级探测游戏窗口：本地客户端 → GFN Chrome 网页版 → GFN 原生客户端。

    返回 (mode, hwnd)。mode 为 GAME_WINDOW_MODE_* 常量之一；
    未找到任何窗口时返回 (GAME_WINDOW_MODE_NOT_FOUND, None)。
    """
    hwnd = find_window_by_process(
        DEFAULT_GAME_PROCESS_NAME,
        selected_hwnd=selected_hwnd,
        last_hwnd=last_hwnd,
    )
    if hwnd:
        return GAME_WINDOW_MODE_NATIVE, hwnd

    hwnd = find_window_by_process(
        GFN_CHROME_PROCESS_NAME,
        hwnd_class=GFN_CHROME_WINDOW_CLASS,
        require_title=True,
        title_regex=GFN_CHROME_TITLE_REGEX,
        selected_hwnd=selected_hwnd,
        last_hwnd=last_hwnd,
    )
    if hwnd:
        return GAME_WINDOW_MODE_GFN_CHROME, hwnd

    hwnd = find_window_by_process(
        GFN_APP_PROCESS_NAME,
        require_title=True,
        title_regex=GFN_APP_TITLE_REGEX,
        selected_hwnd=selected_hwnd,
        last_hwnd=last_hwnd,
    )
    if hwnd:
        return GAME_WINDOW_MODE_GFN_APP, hwnd

    return GAME_WINDOW_MODE_NOT_FOUND, None


# 模块级探测状态（配合独立 reset，遵循仓库模块级状态约定）
_detected_game_mode = None
_detected_game_hwnd = None


def refresh_game_window_mode(selected_hwnd=0, last_hwnd=0):
    """重新探测游戏窗口模式并更新模块级状态。返回 (mode, hwnd)。"""
    global _detected_game_mode, _detected_game_hwnd
    prev = (_detected_game_mode, _detected_game_hwnd)
    mode, hwnd = detect_game_window(selected_hwnd=selected_hwnd, last_hwnd=last_hwnd)
    _detected_game_mode = mode
    _detected_game_hwnd = hwnd
    # 记录窗口类名：用于确认 GFN 客户端等未实测窗口的 class_regex（PRD R3）。
    # 仅在探测结果变化时输出，避免每次任务/缩放触发的重复探测刷屏
    if (mode, hwnd) != prev:
        class_name = get_class_name(hwnd) if hwnd else None
        _log(f"game window mode detected: {mode}, hwnd={hwnd}, class={class_name}")
    return mode, hwnd


def get_game_window_mode(refresh_if_unknown=True):
    """返回最近一次探测到的游戏窗口模式；尚未探测过时可触发一次探测。"""
    if _detected_game_mode is None and refresh_if_unknown:
        mode, _ = refresh_game_window_mode()
        return mode
    return _detected_game_mode


def reset_game_window_mode():
    """清空模块级探测状态，下次 get_game_window_mode 会重新探测。"""
    global _detected_game_mode, _detected_game_hwnd
    _detected_game_mode = None
    _detected_game_hwnd = None


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


def _get_window_work_area(hwnd):
    work_area = _get_monitor_work_area(hwnd)
    if work_area is not None:
        return work_area
    return (
        0,
        0,
        user32.GetSystemMetrics(SM_CXSCREEN),
        user32.GetSystemMetrics(SM_CYSCREEN),
    )


# Window placement within the monitor work area (rcWork excludes the taskbar).
WINDOW_POSITION_CENTER = "center"
WINDOW_POSITION_BOTTOM_RIGHT = "bottom_right"
_WINDOW_POSITIONS = {WINDOW_POSITION_CENTER, WINDOW_POSITION_BOTTOM_RIGHT}


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


def _resolve_window_position(center=None, position=None):
    """Normalize placement. ``position`` wins; legacy ``center`` maps to center/None."""
    if position is not None:
        position = str(position).strip().lower()
        if position not in _WINDOW_POSITIONS:
            raise ValueError(
                f"position must be one of {sorted(_WINDOW_POSITIONS)}, got {position!r}"
            )
        return position
    if center is False:
        return None
    return WINDOW_POSITION_CENTER


def _anchor_point(work_area, window_width, window_height, position):
    """Return (left, top) for ``position`` inside the work area, or None if it won't fit."""
    work_left, work_top, work_right, work_bottom = work_area
    work_width = work_right - work_left
    work_height = work_bottom - work_top
    if work_width < window_width or work_height < window_height:
        return None
    if position == WINDOW_POSITION_BOTTOM_RIGHT:
        # Flush to work-area bottom-right so the taskbar never covers the window.
        return work_right - window_width, work_bottom - window_height
    return (
        work_left + (work_width - window_width) // 2,
        work_top + (work_height - window_height) // 2,
    )


def move_window(hwnd, position=WINDOW_POSITION_CENTER):
    """Move a window within the monitor work area without resizing it."""
    if not hwnd or not position:
        return False
    position = _resolve_window_position(position=position)
    rect = get_window_rect(hwnd)
    if rect is None:
        return False
    left, top, right, bottom = rect
    window_width = right - left
    window_height = bottom - top
    expected = _anchor_point(
        _get_window_work_area(hwnd), window_width, window_height, position
    )
    if expected is None:
        return False
    expected_left, expected_top = expected
    if left == expected_left and top == expected_top:
        return True
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
        left, top, _, _ = rect
        if left == expected_left and top == expected_top:
            return True
        kernel32.Sleep(100)
    return False


def resize_window(hwnd, width, height, center=True, position=None):
    """Resize the outer window, then optionally place it in the work area."""
    if not hwnd:
        return False
    width = int(width)
    height = int(height)
    position = _resolve_window_position(center=center, position=position)
    flags = SWP_SHOWWINDOW | SWP_NOZORDER | SWP_NOMOVE
    if not user32.SetWindowPos(hwnd, None, 0, 0, width, height, flags):
        return False
    kernel32.Sleep(10)

    expected_left = None
    expected_top = None
    if position:
        rect = get_window_rect(hwnd)
        if rect is None:
            return False
        left, top, right, bottom = rect
        window_width = right - left
        window_height = bottom - top
        expected = _anchor_point(
            _get_window_work_area(hwnd), window_width, window_height, position
        )
        if expected is None:
            return False
        expected_left, expected_top = expected
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
        pos_ok = not position or (left == expected_left and top == expected_top)
        if size_ok and pos_ok:
            break
        kernel32.Sleep(100)
    kernel32.Sleep(500)
    return True


def resize_client_area(
    hwnd,
    width,
    height,
    center=True,
    position=None,
    tolerance=2,
    manage_title_bar=True,
):
    """Resize a window so its client area matches the target size.

    manage_title_bar=False 时不强制恢复 WS_CAPTION——GFN 客户端等
    无边框 CEF 窗口按原样缩放（边框差值按实际 0 计算）。
    """
    if not hwnd:
        return False
    target_width = int(width)
    target_height = int(height)
    requested_position = position
    position = _resolve_window_position(center=center, position=position)
    if user32.IsIconic(hwnd) or user32.IsZoomed(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        kernel32.Sleep(100)
    if manage_title_bar:
        show_title_bar(hwnd)

    current_client = get_client_size(hwnd)
    current_rect = get_window_rect(hwnd)
    if current_client is None or current_rect is None:
        return False
    if (
        abs(current_client[0] - target_width) <= tolerance
        and abs(current_client[1] - target_height) <= tolerance
    ):
        if requested_position is not None and position:
            return move_window(hwnd, position)
        return True

    left, top, right, bottom = current_rect
    window_width = right - left
    window_height = bottom - top
    border = max(0, window_width - current_client[0])
    title_height = max(0, window_height - current_client[1])
    resized_width = target_width + border
    resized_height = target_height + title_height
    work_left, work_top, work_right, work_bottom = _get_window_work_area(hwnd)
    work_width = work_right - work_left
    work_height = work_bottom - work_top
    if work_width < resized_width or work_height < resized_height:
        return False

    if not resize_window(
        hwnd, resized_width, resized_height, center=center, position=position
    ):
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
    position=None,
    tolerance=2,
    settle_ms=300,
    hwnd_class=None,
    require_title=False,
    title_regex=None,
    selected_hwnd=0,
    last_hwnd=0,
    manage_title_bar=True,
):
    """Find a process window and resize its client area to the target size."""
    # Keep the caller's explicit position so already-matched windows can still
    # be docked (e.g. GFN bottom-right) without moving local windows that only
    # used the default center-on-resize behavior.
    requested_position = position
    position = _resolve_window_position(center=center, position=position)
    hwnd = find_window_by_process(
        process_name,
        hwnd_class=hwnd_class,
        require_title=require_title,
        title_regex=title_regex,
        selected_hwnd=selected_hwnd,
        last_hwnd=last_hwnd,
    )
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
        if requested_position is not None and position:
            if not move_window(hwnd, position):
                return {
                    "success": False,
                    "reason": "reposition_failed",
                    "hwnd": hwnd,
                    "before": before,
                    "after": before,
                }
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
        position=position,
        tolerance=tolerance,
        manage_title_bar=manage_title_bar,
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
    process_name=None,
    settle_ms=DEFAULT_WINDOW_RESIZE_SETTLE_MS,
    **kwargs,
):
    """Resize the game window client area to the target resolution.

    process_name 为 None 时自动探测运行模式（本地 / GFN Chrome / GFN 原生客户端）
    并路由到对应窗口；显式传入 process_name 则维持旧行为直接按进程名查找。
    返回 dict 额外携带 "mode" 键供调用方区分运行模式。
    """
    if process_name is not None:
        result = ensure_process_client_size(
            process_name,
            width,
            height,
            settle_ms=settle_ms,
            **kwargs,
        )
        result["mode"] = None
        return result

    mode, hwnd = refresh_game_window_mode(
        selected_hwnd=kwargs.get("selected_hwnd", 0),
        last_hwnd=kwargs.get("last_hwnd", 0),
    )

    if mode == GAME_WINDOW_MODE_NOT_FOUND:
        return {
            "success": False,
            "reason": "window_not_found",
            "mode": mode,
            "hwnd": None,
            "before": None,
            "after": None,
        }

    if mode == GAME_WINDOW_MODE_GFN_APP:
        # GFN 原生客户端流窗口为无边框 CEF 窗口，接受标准 MoveWindow/SetWindowPos
        # 缩放（GFNWindowMover 即用此方式）。manage_title_bar=False 保持无边框。
        # 默认贴到工作区右下角（rcWork，避开任务栏），与 GFNWindowMover 布局一致。
        passthrough = {
            k: v
            for k, v in kwargs.items()
            if k not in ("hwnd_class", "require_title", "title_regex", "manage_title_bar")
        }
        passthrough.setdefault("position", WINDOW_POSITION_BOTTOM_RIGHT)
        result = ensure_process_client_size(
            GFN_APP_PROCESS_NAME,
            width,
            height,
            settle_ms=settle_ms,
            require_title=True,
            title_regex=GFN_APP_TITLE_REGEX,
            manage_title_bar=False,
            **passthrough,
        )
        result["mode"] = mode
        if not result.get("success"):
            # 缩放未生效时优雅降级：任务继续运行，由调用方引导用户
            # 在 GFN 设置中固定 720p 串流或使用外部工具调整窗口
            _log(
                f"GFN app window resize failed ({result.get('reason')}), "
                f"client size={result.get('after')}, expected {int(width)}x{int(height)}",
                level="warning",
            )
            result["success"] = True
            result["reason"] = "gfn_app_resize_failed"
        return result

    if mode == GAME_WINDOW_MODE_GFN_CHROME:
        passthrough = {
            k: v
            for k, v in kwargs.items()
            if k not in ("hwnd_class", "require_title", "title_regex")
        }
        result = ensure_process_client_size(
            GFN_CHROME_PROCESS_NAME,
            width,
            height,
            settle_ms=settle_ms,
            hwnd_class=GFN_CHROME_WINDOW_CLASS,
            require_title=True,
            title_regex=GFN_CHROME_TITLE_REGEX,
            **passthrough,
        )
        result["mode"] = mode
        return result

    result = ensure_process_client_size(
        DEFAULT_GAME_PROCESS_NAME,
        width,
        height,
        settle_ms=settle_ms,
        **kwargs,
    )
    result["mode"] = mode
    return result
