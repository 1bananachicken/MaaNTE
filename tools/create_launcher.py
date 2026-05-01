# -*- coding: utf-8 -*-
"""CI 构建脚本：创建 MaaNTE 启动器。

将 MaaNTE.exe 重命名为 MaaNTE_core.exe，
创建 MaaNTE.bat 启动器，在启动前弹出首次使用警告。
"""

import os
import shutil
import sys
from pathlib import Path

install_path = Path(__file__).parent.parent.parent / "install"


def create_launcher():
    maa_exe = install_path / "MaaNTE.exe"
    maa_core = install_path / "MaaNTE_core.exe"

    if not maa_exe.exists():
        print(f"警告: {maa_exe} 不存在，跳过创建启动器")
        return

    # 重命名原始 exe
    shutil.move(str(maa_exe), str(maa_core))
    print(f"已重命名 MaaNTE.exe -> MaaNTE_core.exe")

    # 复制启动器脚本
    launcher_src = Path(__file__).parent / "launcher.py"
    launcher_dst = install_path / "launcher.py"
    if launcher_src.exists():
        shutil.copy2(str(launcher_src), str(launcher_dst))
        print(f"已复制 launcher.py")

    # 创建 bat 启动器
    bat_content = """@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if not exist "config\\warning_shown.json" (
    python\\python.exe launcher.py
) else (
    findstr /c:"true" "config\\warning_shown.json" >nul 2>&1
    if errorlevel 1 (
        python\\python.exe launcher.py
    )
)

start "" "%~dp0MaaNTE_core.exe"
"""

    bat_path = install_path / "MaaNTE.bat"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
    print(f"已创建 MaaNTE.bat")


if __name__ == "__main__":
    create_launcher()
