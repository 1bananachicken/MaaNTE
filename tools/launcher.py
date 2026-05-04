# -*- coding: utf-8 -*-
"""首次使用警告弹窗启动器。

在 MFAAvalonia 启动前检查并显示警告弹窗，
然后启动主程序 MaaNTE.exe。
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def show_warning():
    """显示首次使用警告弹窗（5秒自动关闭）。"""
    _WARNING_TEXT = (
        "欢迎使用 MaaNTE\n"
        "\n"
        "MaaNTE 为免费开源项目，从未授权任何人以任何形式进行售卖。\n"
        "  - 如在闲鱼、淘宝等平台购买了本软件，请立即申请退款并举报商家\n"
        "  - 可凭此弹窗截图要求退款，维护自身权益\n"
        "  - 你付给倒卖者的每一分钱都会让开源社区更艰难\n"
        "\n"
        "Mirror酱 是我们的合作伙伴，提供下载加速服务，不属于售卖行为\n"
        "\n"
        "—————————————————————————————\n"
        "\n"
        "本软件开源免费，仅供学习交流使用。\n"
        "使用本软件产生的所有后果由使用者自行承担，与开发者团队无关。\n"
        "开发者团队拥有本项目的最终解释权。"
    )

    try:
        import ctypes

        MB_OK = 0x00000000
        MB_TOPMOST = 0x00040000
        MB_ICONWARNING = 0x00000030
        MB_SETFOREGROUND = 0x00010000
        TIMEOUT_MS = 5000

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.MessageBoxTimeoutW(
            hwnd,
            _WARNING_TEXT,
            "MaaNTE - 首次使用须知",
            MB_OK | MB_TOPMOST | MB_ICONWARNING | MB_SETFOREGROUND,
            0,
            TIMEOUT_MS,
        )
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                _WARNING_TEXT,
                "MaaNTE - 首次使用须知",
                0x00000040 | 0x00040000,
            )
        except Exception:
            pass


def main():
    work_dir = Path(__file__).resolve().parent
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
        show_warning()
        try:
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"shown": True}, f, ensure_ascii=False)
        except Exception:
            pass

    maa_exe = work_dir / "MaaNTE.exe"
    if maa_exe.exists():
        subprocess.Popen([str(maa_exe)], cwd=str(work_dir))
    else:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                "找不到 MaaNTE.exe，请检查安装是否完整。",
                "MaaNTE",
                0x00000010,
            )
        except Exception:
            print("错误：找不到 MaaNTE.exe")


if __name__ == "__main__":
    main()
