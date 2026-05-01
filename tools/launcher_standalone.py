# -*- coding: utf-8 -*-
"""MaaNTE 启动器 - 独立编译版"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

WARN_TEXT = (
    "欢迎使用 MaaNTE\r\n\r\n"
    "MaaNTE 为免费开源项目，从未授权任何人以任何形式进行售卖。\r\n"
    "  - 如在闲鱼、淘宝等平台购买了本软件，请立即申请退款并举报商家\r\n"
    "  - 可凭此弹窗截图要求退款，维护自身权益\r\n"
    "  - 你付给倒卖者的每一分钱都会让开源社区更艰难\r\n\r\n"
    "Mirror酱 是我们的合作伙伴，提供下载加速服务，不属于售卖行为\r\n\r\n"
    "───────────────────────────\r\n\r\n"
    "本软件开源免费，仅供学习交流使用。\r\n"
    "使用本软件产生的所有后果由使用者自行承担，与开发者团队无关。\r\n"
    "开发者团队拥有本项目的最终解释权。"
)


def main():
    work_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    os.chdir(work_dir)

    config_path = work_dir / "config" / "warning_shown.json"
    shown = False
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                shown = json.load(f).get("shown", False)
        except Exception:
            pass

    if not shown:
        MB_OK = 0x0
        MB_TOPMOST = 0x40000
        MB_ICONWARNING = 0x30
        MB_SETFOREGROUND = 0x10000

        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            pMsgBox = ctypes.windll.user32.MessageBoxTimeoutW
            pMsgBox(hwnd, WARN_TEXT, "MaaNTE - 首次使用须知",
                    MB_OK | MB_TOPMOST | MB_ICONWARNING | MB_SETFOREGROUND,
                    0, 5000)
        except Exception:
            ctypes.windll.user32.MessageBoxW(
                None, WARN_TEXT, "MaaNTE - 首次使用须知",
                MB_OK | MB_TOPMOST | MB_ICONWARNING)

        try:
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"shown": True}, f, ensure_ascii=False)
        except Exception:
            pass

    core = work_dir / "MaaNTE_core.exe"
    if core.exists():
        subprocess.Popen([str(core)], cwd=str(work_dir))
    else:
        ctypes.windll.user32.MessageBoxW(
            None, "找不到 MaaNTE_core.exe，请检查安装是否完整。",
            "MaaNTE", 0x10)


if __name__ == "__main__":
    main()
