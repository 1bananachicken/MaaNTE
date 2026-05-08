import time
import win32gui
import win32con
import win32api

# ==========================================
# 填入游戏的精确名称
WINDOW_TITLE = "异环  "  
# ==========================================

class MaaKeyboardBridge:
    def __init__(self, controller=None, hold_seconds: float = 0.008, wait_jobs: bool = False):
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        self.hold_seconds = hold_seconds
        
        if not self.hwnd:
            print(f"【警告】找不到遊戲視窗: {WINDOW_TITLE}")
            
        self.mapping = {
            60: (0x5A, None), 61: (0x5A, 'shift'), 62: (0x58, None), 63: (0x43, 'ctrl'), 64: (0x43, None),
            65: (0x56, None), 66: (0x56, 'shift'), 67: (0x42, None), 68: (0x42, 'shift'), 69: (0x4E, None),
            70: (0x4D, 'ctrl'), 71: (0x4D, None),
            72: (0x41, None), 73: (0x41, 'shift'), 74: (0x53, None), 75: (0x44, 'ctrl'), 76: (0x44, None),
            77: (0x46, None), 78: (0x46, 'shift'), 79: (0x47, None), 80: (0x47, 'shift'), 81: (0x48, None),
            82: (0x4A, 'ctrl'), 83: (0x4A, None),
            84: (0x51, None), 85: (0x51, 'shift'), 86: (0x57, None), 87: (0x45, 'ctrl'), 88: (0x45, None),
            89: (0x52, None), 90: (0x52, 'shift'), 91: (0x54, None), 92: (0x54, 'shift'), 93: (0x59, None),
            94: (0x55, 'ctrl'), 95: (0x55, None)
        }

    def _force_send_key(self, vk_code, is_down):
        """稳定后台发送"""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return

        scan_code = win32api.MapVirtualKey(vk_code, 0)
        # 強制視窗訊息隊列 (骗过后台检测)
        win32gui.SendMessage(self.hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)
        
        if is_down:
            lparam = 1 | (scan_code << 16)
            win32gui.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, lparam)
        else:
            lparam = 1 | (scan_code << 16) | 0xC0000000
            win32gui.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, lparam)

    def execute_chord(self, midi_notes):
        """融合：分组并发，秒杀黑乐谱延迟"""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return

        normal_keys, shift_keys, ctrl_keys = [], [], []

        for note in midi_notes:
            if note in self.mapping:
                vk_code, mod = self.mapping[note]
                if mod == 'shift': shift_keys.append(vk_code)
                elif mod == 'ctrl': ctrl_keys.append(vk_code)
                else: normal_keys.append(vk_code)

        # 严格隔离分组弹奏
        self._press_group(normal_keys, None)
        # 使用 0xA0 (左Shift) 和 0xA2 (左Ctrl) 解决游戏跑调问题
        self._press_group(shift_keys, 0xA0) 
        self._press_group(ctrl_keys, 0xA2)

    def _press_group(self, vk_codes, mod_vk):
        if not vk_codes:
            return

        # 1. 按下修饰键
        if mod_vk:
            self._force_send_key(mod_vk, True)
            time.sleep(0.002) # 给游戏引擎反应时间，防跑调

        # 2. 批量按下主按键
        for vk in vk_codes:
            self._force_send_key(vk, True)

        # 3. 极速停留 (默认 0.008s)
        if self.hold_seconds > 0:
            time.sleep(self.hold_seconds)

        # 4. 批量抬起主按键
        for vk in reversed(vk_codes):
            self._force_send_key(vk, False)

        # 5. 抬起修饰键
        if mod_vk:
            time.sleep(0.001)
            self._force_send_key(mod_vk, False)
