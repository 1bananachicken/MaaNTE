# -*- coding: utf-8 -*-
"""CI 构建脚本：编译并安装 MaaNTE 启动器。

将 MaaNTE.exe 重命名为 MaaNTE_core.exe，
用 PyInstaller 将 launcher_standalone.py 编译为新的 MaaNTE.exe。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def _safe_reconfigure(stream):
    """安全地重配置流编码，兼容旧 Python 或被替换的流对象。"""
    fn = getattr(stream, "reconfigure", None)
    if callable(fn):
        fn(encoding="utf-8")


_safe_reconfigure(sys.stdout)
_safe_reconfigure(sys.stderr)

script_dir = Path(__file__).parent
install_path = script_dir.parent.parent / "install"


def create_launcher():
    maa_exe = install_path / "MaaNTE.exe"
    maa_core = install_path / "MaaNTE_core.exe"

    if not maa_exe.exists():
        print(f"警告: {maa_exe} 不存在，跳过创建启动器")
        return

    # 重命名原始 exe
    try:
        shutil.move(str(maa_exe), str(maa_core))
        print(f"已重命名 MaaNTE.exe -> MaaNTE_core.exe")
    except Exception as e:
        print(f"错误: 无法重命名 {maa_exe} -> {maa_core}: {e}")
        return

    launcher_src = script_dir / "launcher_standalone.py"
    if not launcher_src.exists():
        print(f"错误: {launcher_src} 不存在")
        return

    # 用 PyInstaller 编译
    timeout_env = os.environ.get("PYINSTALLER_TIMEOUT")
    try:
        pyinstaller_timeout = int(timeout_env) if timeout_env else 300
    except ValueError:
        pyinstaller_timeout = 300

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "PyInstaller",
                "--onefile", "--noconsole",
                "--name", "MaaNTE",
                "--distpath", str(install_path),
                "--workpath", str(script_dir / "_build"),
                "--specpath", str(script_dir),
                "--clean", "-y",
                str(launcher_src),
            ],
            capture_output=True, text=True, timeout=pyinstaller_timeout,
        )
        if result.returncode == 0:
            print("已编译 MaaNTE.exe 启动器 (PyInstaller)")
            # 清理构建产物
            shutil.rmtree(script_dir / "_build", ignore_errors=True)
            spec_file = script_dir / "MaaNTE.spec"
            spec_file.unlink(missing_ok=True)
            return
        else:
            print(f"PyInstaller 编译失败:\n{result.stderr}")
    except Exception as e:
        print(f"PyInstaller 异常: {e}")

    # 降级：恢复原始 exe
    shutil.move(str(maa_core), str(maa_exe))
    print("编译失败，已恢复原始 MaaNTE.exe")


if __name__ == "__main__":
    create_launcher()
