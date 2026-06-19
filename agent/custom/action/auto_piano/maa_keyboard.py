import time
import ctypes
from .key_mapping import NOTE_KEY_MAPPING
from utils.logger import logger

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
# 填入游戏的精确名称列表（脚本会按从上到下的顺序匹配）
WINDOW_TITLES = [
    "NTE  ",
    "异环  ",
]
# ==========================================


def get_lparam(vk_code, is_down=True, repeat=False):
    """构建底层硬件扫描码"""
    scan_code = user32.MapVirtualKeyW(vk_code, 0)
    lparam = 1 | (scan_code << 16)
    if is_down and repeat:
        lparam |= 0x40000000
    if not is_down:
        lparam |= 0xC0000000
    return lparam


class MaaKeyboardBridge:
    def __init__(
        self,
        controller=None,
        hold_seconds: float = 0.008,
        wait_jobs: bool = False,
        mapping: dict | None = None,
        sustain_mode: str = "repeat",
        repeat_interval: float = 0.045,
    ):
        self.controller = controller
        self.mapping = mapping if mapping is not None else NOTE_KEY_MAPPING
        self.hold_seconds = hold_seconds
        self.wait_jobs = wait_jobs
        self.sustain_mode = sustain_mode if sustain_mode in ("repeat", "hold") else "repeat"
        self.repeat_interval = min(max(float(repeat_interval), 0.015), 0.2)
        self.hwnd = 0

        # 跟踪当前按下的音符引用计数和修饰键引用计数
        self._active_counts: dict[int, int] = {}  # midi_note -> ref_count
        self._active_modifiers: dict[int, int] = {}  # vk -> ref_count
        self._last_refresh = 0.0

        if self.controller is not None:
            logger.info(
                "自动钢琴使用 Maa controller 发送按键（延音模式: %s, 补发间隔: %.0fms）",
                self.sustain_mode,
                self.repeat_interval * 1000,
            )

        if self.controller is None or self.sustain_mode == "repeat":
            # 按顺序遍历列表，寻找第一个存在的游戏窗口
            for title in WINDOW_TITLES:
                hwnd = user32.FindWindowW(None, title)
                if hwnd:
                    self.hwnd = hwnd
                    logger.info("已连接到游戏窗口: '%s' (HWND: %s)", title, self.hwnd)
                    break  # 找到了就立刻停止搜索

            if not self.hwnd:
                logger.warning("未找到列表中的任何窗口，请检查游戏是否运行！列表: %s", WINDOW_TITLES)

    # ------------------------------------------------------------------
    # 底层发送
    # ------------------------------------------------------------------
    def _force_send_key(self, vk_code, is_down, repeat=False):
        """带有强行唤醒的底层发送器"""
        if not self.hwnd:
            return

        # 发送前给窗口发一个 WM_ACTIVATE 信号，骗过后台检测。
        # 使用 PostMessageW（异步）避免阻塞，且每 50ms 最多发一次防止消息堆积。
        now = time.perf_counter()
        if now - getattr(self, "_last_activate", 0) > 0.05:
            user32.PostMessageW(self.hwnd, WM_ACTIVATE, WA_CLICKACTIVE, 0)
            self._last_activate = now

        lparam = get_lparam(vk_code, is_down, repeat=repeat)
        msg = WM_KEYDOWN if is_down else WM_KEYUP
        user32.PostMessageW(self.hwnd, msg, vk_code, lparam)

    def _send_key(self, vk_code, is_down, repeat=False):
        """优先通过 Maa controller 发送按键，失败时回退到窗口消息。"""
        if self.controller is not None:
            try:
                job = (
                    self.controller.post_key_down(vk_code)
                    if is_down
                    else self.controller.post_key_up(vk_code)
                )
                if self.wait_jobs and job is not None:
                    job.wait()
                return
            except Exception:
                logger.exception("Maa controller 发送按键失败，回退到 PostMessage")
                self.controller = None

        self._force_send_key(vk_code, is_down, repeat=repeat)

    @staticmethod
    def _parse_action(action: str) -> tuple[str | None, str]:
        """解析键位动作字符串，例如 'shift+z' -> ('shift', 'z')"""
        if "+" in action:
            mod, key = action.split("+", 1)
            return mod, key
        return None, action

    # ------------------------------------------------------------------
    # 单音符控制（支持延音）
    # ------------------------------------------------------------------
    def press_note(self, midi_note: int) -> None:
        """按下一个音符（引用计数管理，同一音符多次按下不会重复发送 key down）。"""
        if midi_note not in self.mapping or (self.controller is None and not self.hwnd):
            return

        action = self.mapping[midi_note]
        modifier, key = self._parse_action(action)

        vk = WIN32_VK.get(key)
        if vk is None:
            return

        # 引用计数 +1
        self._active_counts[midi_note] = self._active_counts.get(midi_note, 0) + 1
        if self._active_counts[midi_note] > 1:
            # 已经有其他声部在按这个 MIDI 音符，不需要重复发送 key down
            return

        mod_vk = WIN32_VK.get(modifier) if modifier else None

        # 检查是否有其他音符使用同一个物理键（base key）
        other_using_same_key = None
        other_mod_vk = None
        for other_note, count in self._active_counts.items():
            if other_note != midi_note and count > 0:
                other_action = self.mapping[other_note]
                other_mod, other_key = self._parse_action(other_action)
                if other_key == key:
                    other_using_same_key = other_note
                    other_mod_vk = WIN32_VK.get(other_mod) if other_mod else None
                    break

        if other_using_same_key is not None:
            # 同一个 base key 上已经有其他音符在按着
            # 只需要调整 modifier（如果有变化），不需要重复发送 key down
            if other_mod_vk != mod_vk:
                # 释放旧的 modifier
                if other_mod_vk and other_mod_vk in self._active_modifiers:
                    self._active_modifiers[other_mod_vk] -= 1
                    if self._active_modifiers[other_mod_vk] <= 0:
                        self._send_key(other_mod_vk, False)
                        self._active_modifiers[other_mod_vk] = 0
                # 按下新的 modifier
                if mod_vk:
                    if mod_vk not in self._active_modifiers or self._active_modifiers[mod_vk] == 0:
                        self._send_key(mod_vk, True)
                    self._active_modifiers[mod_vk] = self._active_modifiers.get(mod_vk, 0) + 1
            return

        # 正常流程：base key 没有被其他音符使用
        if mod_vk:
            if mod_vk not in self._active_modifiers or self._active_modifiers[mod_vk] == 0:
                self._send_key(mod_vk, True)
            self._active_modifiers[mod_vk] = self._active_modifiers.get(mod_vk, 0) + 1

        self._send_key(vk, True)

    def release_note(self, midi_note: int) -> None:
        """抬起一个音符（引用计数管理，只有当所有声部都释放后才真正发送 key up）。"""
        if midi_note not in self._active_counts or midi_note not in self.mapping:
            return

        # 引用计数 -1
        self._active_counts[midi_note] -= 1
        if self._active_counts[midi_note] > 0:
            # 还有其他声部在按这个 MIDI 音符，不要发送 key up
            return

        del self._active_counts[midi_note]

        action = self.mapping[midi_note]
        modifier, key = self._parse_action(action)

        vk = WIN32_VK.get(key)
        if vk is None:
            return

        # 检查是否有其他活动音符使用同一个物理键
        # （例如 C=z 和 C#=shift+z 共享 z 键，释放 C 时不应该发送 z up）
        for other_note in self._active_counts:
            other_action = self.mapping[other_note]
            other_mod, other_key = self._parse_action(other_action)
            if other_key == key:
                # 有其他音符在使用同一个物理键，不释放 base key
                # 但需要释放当前音符的 modifier（如果有），防止泄漏
                mod_vk = WIN32_VK.get(modifier) if modifier else None
                if mod_vk and mod_vk in self._active_modifiers and self._active_modifiers[mod_vk] > 0:
                    self._active_modifiers[mod_vk] -= 1
                    if self._active_modifiers[mod_vk] <= 0:
                        self._send_key(mod_vk, False)
                        self._active_modifiers[mod_vk] = 0
                return

        self._send_key(vk, False)

        # 修饰键引用计数 -1，归零时抬起
        mod_vk = WIN32_VK.get(modifier) if modifier else None
        if mod_vk and mod_vk in self._active_modifiers and self._active_modifiers[mod_vk] > 0:
            self._active_modifiers[mod_vk] -= 1
            if self._active_modifiers[mod_vk] <= 0:
                self._send_key(mod_vk, False)
                self._active_modifiers[mod_vk] = 0

    def release_all(self) -> None:
        """抬起所有当前按下的音符（安全清理）。"""
        for note in list(self._active_counts.keys()):
            self._active_counts[note] = 1
            self.release_note(note)

    def refresh_active_keys(self, force: bool = False) -> None:
        """刷新仍在延音中的按键，避免游戏窗口长按状态超时丢失。"""
        if self.sustain_mode != "repeat":
            return
        if not self.hwnd or not self._active_counts:
            return

        now = time.perf_counter()
        if not force and now - self._last_refresh < self.repeat_interval:
            return
        self._last_refresh = now

        for vk, count in list(self._active_modifiers.items()):
            if count > 0:
                self._force_send_key(vk, True, repeat=True)

        base_keys: set[int] = set()
        for note, count in list(self._active_counts.items()):
            if count <= 0 or note not in self.mapping:
                continue
            _, key = self._parse_action(self.mapping[note])
            vk = WIN32_VK.get(key)
            if vk is not None:
                base_keys.add(vk)

        for vk in base_keys:
            self._force_send_key(vk, False)
            self._force_send_key(vk, True)

    # ------------------------------------------------------------------
    # 向后兼容：和弦一次性播放（旧逻辑，现在 player 不再调用）
    # ------------------------------------------------------------------
    def execute_chord(self, midi_notes):
        if self.controller is None and not self.hwnd:
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
