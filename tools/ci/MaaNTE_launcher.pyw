import ctypes
import subprocess
import sys
import os
import tempfile
import shutil

SW_SHOWNORMAL = 1
MB_YESNO = 0x04
MB_ICONWARNING = 0x30
IDYES = 6

BUNDLE_EXE = "MaaNTE-app.exe"
EXTRACT_DIR = os.path.join(tempfile.gettempdir(), "MaaNTE_app")


def get_bundle_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def ensure_extracted():
    src = os.path.join(get_bundle_dir(), BUNDLE_EXE)
    dst = os.path.join(EXTRACT_DIR, BUNDLE_EXE)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def main():
    app = ensure_extracted()

    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin = True

    if not is_admin:
        try:
            result = ctypes.windll.user32.MessageBoxW(
                None,
                "当前未以管理员权限运行，部分功能可能无法正常使用。\n\n是否立即以管理员身份重新启动？",
                "MaaNTE - 权限不足",
                MB_YESNO | MB_ICONWARNING,
            )
        except Exception:
            result = 0

        if result == IDYES:
            try:
                ret = ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", app, None, None, SW_SHOWNORMAL,
                )
                if ret > 32:
                    sys.exit(0)
            except Exception:
                pass

    subprocess.Popen([app])
    sys.exit(0)


if __name__ == "__main__":
    main()
