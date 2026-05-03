# -*- coding: utf-8 -*-
"""MaaNTE 启动器 - 独立编译版

功能：
1. 首次使用弹出警告窗口
2. 校验 interface.json welcome 字段完整性
"""

import ctypes
import hashlib
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

_EXPECTED_WELCOME_HASH = "7b4e40b09fb2eb391beb9943cf491129934abe04cb43eaae5b6965be464773ea"

MB_OK = 0x0
MB_ICONWARNING = 0x30
MB_ICONERROR = 0x10
MB_TOPMOST = 0x40000
MB_SETFOREGROUND = 0x10000
MB_SYSTEMMODAL = 0x1000


def _msgbox(text, title, flags):
    ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def check_first_use(work_dir):
    config_path = work_dir / "config" / "warning_shown.json"
    shown = False
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                shown = json.load(f).get("shown", False)
        except Exception:
            pass

    if not shown:
        _msgbox(
            WARN_TEXT,
            "MaaNTE - 首次使用须知",
            MB_OK | MB_ICONWARNING | MB_TOPMOST | MB_SETFOREGROUND | MB_SYSTEMMODAL,
        )
        try:
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"shown": True}, f, ensure_ascii=False)
        except Exception:
            pass


def check_integrity(work_dir):
    interface_path = work_dir / "interface.json"
    if not interface_path.exists():
        return

    try:
        with open(interface_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        welcome = data.get("welcome", "")
        if not welcome:
            raise ValueError("welcome 字段为空或已被移除")
        actual = hashlib.sha256(welcome.encode("utf-8")).hexdigest()
        if actual != _EXPECTED_WELCOME_HASH:
            raise ValueError(f"哈希不匹配 (期望 {_EXPECTED_WELCOME_HASH[:16]}..., 实际 {actual[:16]}...)")
    except Exception as e:
        _msgbox(
            f"警告内容完整性校验失败！\n\n"
            f"本软件为免费开源项目，从未授权任何人售卖。\n"
            f"如在第三方平台购买了本软件，请立即申请退款并举报。\n\n"
            f"校验详情: {e}",
            "MaaNTE - 完整性校验失败",
            MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND | MB_SYSTEMMODAL,
        )


def main():
    work_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    os.chdir(work_dir)

    check_first_use(work_dir)
    check_integrity(work_dir)

    core = work_dir / "bin" / "MaaNTE_core.exe"
    if core.exists():
        subprocess.Popen([str(core)], cwd=str(work_dir))
    else:
        _msgbox(
            "找不到 bin/MaaNTE_core.exe，请检查安装是否完整。",
            "MaaNTE",
            MB_OK | MB_ICONERROR,
        )


if __name__ == "__main__":
    main()
