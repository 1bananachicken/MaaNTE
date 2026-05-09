import ctypes
import subprocess
import sys
import os

SW_SHOWNORMAL = 1
MB_YESNO = 0x04
MB_ICONWARNING = 0x30
IDYES = 6


def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    app = os.path.join(get_app_dir(), "MaaNTE-app.exe")

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
