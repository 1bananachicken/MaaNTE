# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
from pathlib import Path

# utf-8
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# 获取当前main.py路径并设置上级目录为工作目录
current_file_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_file_path)  # 包含此脚本的目录
project_root_dir = os.path.dirname(current_script_dir)  # 假定的项目根目录

# 更改CWD到项目根目录
if os.getcwd() != project_root_dir:
    os.chdir(project_root_dir)
print(f"set cwd: {os.getcwd()}")

# 将脚本自身的目录添加到sys.path，以便导入utils、maa等模块
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

from utils import logger

VENV_NAME = ".venv"  # 虚拟环境目录的名称
VENV_DIR = Path(project_root_dir) / VENV_NAME

# -----
# region 虚拟环境
# -----


def _is_running_in_our_venv():
    """检查脚本是否在虚拟环境中运行。"""
    # 使用 sys.prefix 和 sys.base_prefix 来判断是否在虚拟环境中
    in_venv = sys.prefix != sys.base_prefix

    if in_venv:
        logger.debug(f"当前在虚拟环境中运行: {sys.prefix}")
    else:
        logger.debug(f"当前不在虚拟环境中，使用系统Python: {sys.prefix}")

    return in_venv


def ensure_venv_and_relaunch_if_needed():
    """
    确保venv存在，并且如果尚未在脚本管理的venv中运行，
    则在其中重新启动脚本。支持Linux和Windows系统。
    """
    logger.info(f"检测到系统: {sys.platform}。当前Python解释器: {sys.executable}")

    if _is_running_in_our_venv():
        logger.info(f"已在目标虚拟环境 ({VENV_DIR}) 中运行。")
        return

    if not VENV_DIR.exists():
        logger.info(f"正在 {VENV_DIR} 创建虚拟环境...")
        try:
            # 使用当前运行此脚本的Python（系统/外部Python）
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                check=True,
                capture_output=True,
            )
            logger.info(f"创建成功")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"创建失败: {e.stderr.decode(errors='ignore') if e.stderr else e.stdout.decode(errors='ignore')}"
            )
            logger.error("正在退出")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(
                f"命令 '{sys.executable} -m venv' 未找到。请确保 'venv' 模块可用。"
            )
            logger.error("无法在没有虚拟环境的情况下继续。正在退出。")
            sys.exit(1)

    if sys.platform.startswith("win"):
        python_in_venv = VENV_DIR / "Scripts" / "python.exe"
    else:
        python3_path = VENV_DIR / "bin" / "python3"
        python_path = VENV_DIR / "bin" / "python"
        if python3_path.exists():
            python_in_venv = python3_path
        elif python_path.exists():
            python_in_venv = python_path
        else:
            python_in_venv = python3_path  # 默认使用python3，让后续错误处理捕获

    if not python_in_venv.exists():
        logger.error(f"在虚拟环境 {python_in_venv} 中未找到Python解释器。")
        logger.error("虚拟环境创建可能失败或虚拟环境结构异常。")
        sys.exit(1)

    logger.info(f"正在使用虚拟环境Python重新启动")

    try:
        # Use absolute path to this script when relaunching inside the venv.
        # sys.argv[0] may be a relative path (e.g. './../agent/main.py') which
        # resolves differently when cwd changes. Use the absolute path of
        # the currently running file (`current_file_path`) to avoid that.
        script_abs = current_file_path
        args = sys.argv[1:]
        cmd = [str(python_in_venv), str(script_abs)] + args
        logger.info(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            env=os.environ.copy(),
            check=False,  # 不在非零退出码时抛出异常
        )
        # 退出时使用子进程的退出码
        sys.exit(result.returncode)

    except Exception as e:
        logger.exception(f"在虚拟环境中重新启动脚本失败: {e}")
        sys.exit(1)


# -----
# region 配置相关
# -----


def read_config(config_name: str, default_config: dict) -> dict:
    """
    通用配置文件读取函数

    Args:
        config_name: 配置文件名（不含.json后缀）
        default_config: 默认配置字典

    Returns:
        配置字典
    """
    config_dir = Path("./config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f"{config_name}.json"

    if not config_path.exists():
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception:
            logger.debug(f"无法写入 {config_name}.json，使用默认配置")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception(f"读取 {config_name}.json 失败，使用默认配置")
        return default_config


def read_interface_version(interface_file_name="./interface.json") -> str:
    interface_path = Path(project_root_dir) / interface_file_name
    assets_interface_path = Path(project_root_dir) / "assets" / interface_file_name

    target_path = None
    if interface_path.exists():
        target_path = interface_path
    elif assets_interface_path.exists():
        return "DEBUG"

    if target_path is None:
        logger.warning("未找到interface.json")
        return "unknown"

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            interface_data = json.load(f)
            return interface_data.get("version", "unknown")
    except Exception:
        logger.exception(f"读取interface.json版本失败，文件路径：{target_path}")
        return "unknown"


def read_pip_config() -> dict:
    default_config = {
        "enable_pip_install": True,
        "mirror": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "backup_mirror": "https://mirrors.ustc.edu.cn/pypi/simple",
    }
    return read_config("pip_config", default_config)


def read_hot_update_config() -> dict:
    """
    读取热更配置
    """
    default_config = {"enable_hot_update": True}
    return read_config("hot_update", default_config)


def _format_env_value(value: str, limit: int = 300) -> str:
    if not value:
        return "<empty>"
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(truncated, total={len(value)})"


def log_pi_environment() -> None:
    pi_env_keys = [
        "PI_INTERFACE_VERSION",
        "PI_CLIENT_NAME",
        "PI_CLIENT_VERSION",
        "PI_CLIENT_LANGUAGE",
        "PI_CLIENT_MAAFW_VERSION",
        "PI_VERSION",
        "PI_CONTROLLER",
        "PI_RESOURCE",
    ]

    logger.debug("PI environment snapshot:")
    for key in pi_env_keys:
        logger.debug(f"{key}={_format_env_value(os.getenv(key, ''))}")


# -----
# region 依赖安装
# -----


def find_local_wheels_dir():
    """查找本地deps目录中的whl文件"""
    project_root = Path(project_root_dir)
    deps_dir = project_root / "deps"

    if deps_dir.exists() and any(deps_dir.glob("*.whl")):
        whl_count = len(list(deps_dir.glob("*.whl")))
        logger.debug(f"发现本地deps目录包含 {whl_count} 个 whl 文件")
        return deps_dir

    logger.debug("未找到deps目录或目录中无 whl 文件")
    return None


def _run_pip_command(cmd_args: list, operation_name: str) -> bool:
    try:
        logger.info(f"开始 {operation_name}")
        logger.debug(f"执行命令: {' '.join(cmd_args)}")

        # 使用subprocess.Popen进行实时输出
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # 行缓冲
            universal_newlines=True,
        )

        # 收集所有输出用于日志记录
        all_output = []

        # 实时读取并显示输出
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip("\n\r")
                if line.strip():  # 只显示非空行
                    all_output.append(line)  # 收集到列表中

        # 等待进程结束
        return_code = process.wait()

        # 记录完整输出到日志
        if all_output:
            full_output = "\n".join(all_output)
            logger.debug(f"{operation_name} 输出:\n{full_output}")

        if return_code == 0:
            logger.info(f"{operation_name} 完成")
            return True
        else:
            logger.error(f"{operation_name} 时出错。返回码: {return_code}")
            return False

    except Exception as e:
        logger.exception(f"{operation_name} 时发生未知异常: {e}")
        return False


def install_requirements(
    req_file="requirements.txt", pip_config: dict | None = None
) -> bool:
    req_path = Path(project_root_dir) / req_file  # 确保相对于项目根目录
    if not req_path.exists():
        logger.error(f"{req_file} 文件不存在于 {req_path.resolve()}")
        return False

    # 查找本地deps目录
    deps_dir = find_local_wheels_dir()
    if deps_dir:
        logger.debug(f"使用本地 whl 文件安装，目录: {deps_dir}")

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "-r",
            str(req_path),
            "--no-warn-script-location",
            "--break-system-packages",
            "--find-links",
            str(deps_dir),  # pip会优先使用这里的文件
            "--no-index",  # 禁止在线索引
        ]

        if _run_pip_command(cmd, f"从本地deps安装依赖"):
            return True
        else:
            logger.warning("本地deps安装失败，回退到纯在线安装")

    # 回退到在线安装
    primary_mirror = pip_config.get("mirror", "") if pip_config else ""
    backup_mirror = pip_config.get("backup_mirror", "") if pip_config else ""

    if primary_mirror:
        # 使用主镜像源，只添加一个备用源避免冲突
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "-r",
            str(req_path),
            "--no-warn-script-location",
            "--break-system-packages",
            "-i",
            primary_mirror,
        ]

        # 只添加一个备用源
        if backup_mirror:
            cmd.extend(["--extra-index-url", backup_mirror])
            logger.info(f"使用主源 {primary_mirror} 和备用源 {backup_mirror} 安装依赖")
        else:
            logger.info(f"使用主源 {primary_mirror} 安装依赖")

        if _run_pip_command(cmd, f"从 {req_path.name} 安装依赖"):
            return True
        else:
            logger.error("在线安装失败")
            return False
    else:
        # 如果没有配置主镜像源，使用pip的本地全局配置
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "-r",
            str(req_path),
            "--no-warn-script-location",
            "--break-system-packages",
        ]

        if _run_pip_command(cmd, f"从 {req_path.name} 安装依赖 (本地全局配置)"):
            return True
        else:
            logger.error("使用pip本地全局配置安装失败")
            return False


def check_and_install_dependencies():
    """检查并安装项目依赖"""
    pip_config = read_pip_config()
    enable_pip_install = pip_config.get("enable_pip_install", True)

    logger.debug(f"启用 pip 安装依赖: {enable_pip_install}")

    if enable_pip_install:
        logger.info("开始安装/更新依赖")
        if install_requirements(pip_config=pip_config):
            logger.info("依赖检查和安装完成")
        else:
            logger.warning("依赖安装失败，程序可能无法正常运行")
    else:
        logger.info("Pip 依赖安装已禁用，跳过依赖安装")


# -----
# region 环境检测
# -----


def _check_chinese_path():
    """检测当前程序是否运行在含中文字符的目录下"""
    import re

    cwd = os.getcwd()
    if re.search(r"[一-鿿]", cwd):
        logger.warning(
            f"当前运行目录含中文字符: {cwd}\n"
            "部分组件对中文路径兼容性较差，请将程序移动至纯英文路径下运行。"
        )


def _check_admin_privilege():
    """检查是否以管理员权限运行，若否则输出警告"""
    import ctypes

    if ctypes.windll.shell32.IsUserAnAdmin():
        return

    logger.warning(
        "未以管理员权限运行，部分输入功能可能无法正常使用。"
        "请右键 MaaNTE.exe → 以管理员身份运行。"
    )


def _check_game_resolution():
    """连接控制器后检测游戏窗口分辨率"""
    from utils.win32_process import find_window_by_process, get_client_size

    hwnd = find_window_by_process("HTGame.exe")
    if hwnd is None:
        logger.warning("分辨率检测: 未找到游戏窗口 (HTGame.exe)")
        return

    size = get_client_size(hwnd)
    if size is None:
        logger.warning("分辨率检测: 无法获取窗口尺寸")
        return

    w, h = size
    if (w, h) == (1280, 720):
        logger.info(f"当前窗口分辨率: {w}x{h} [正常]")
    else:
        logger.warning(
            f"当前窗口分辨率: {w}x{h}，请使用 1280x720 分辨率。"
            "请将游戏设置为 1280x720 窗口化模式，否则部分功能可能异常。"
        )


# -----
# region 核心业务
# -----


def agent(is_dev_mode=False):
    try:
        # 清理模块缓存
        utils_modules = [
            name for name in list(sys.modules.keys()) if name.startswith("utils")
        ]
        for module_name in utils_modules:
            del sys.modules[module_name]

        # 动态导入 utils 的所有内容
        import utils
        import importlib

        importlib.reload(utils)

        # 将 utils 的所有公共属性导入到当前命名空间
        for attr_name in dir(utils):
            if not attr_name.startswith("_"):
                globals()[attr_name] = getattr(utils, attr_name)

        if is_dev_mode:
            from utils.logger import change_console_level

            change_console_level("DEBUG")
            logger.info("开发模式：日志等级已设置为DEBUG")
            

        from maa.agent.agent_server import AgentServer
        from maa.tasker import Tasker

        import custom

        Tasker.set_log_dir("./debug")

        if len(sys.argv) < 2:
            logger.error("缺少必要的 socket_id 参数")
            return

        socket_id = sys.argv[-1]
        logger.debug(f"socket_id: {socket_id}")

        log_pi_environment()
        AgentServer.start_up(socket_id)
        logger.info("AgentServer启动")
        _check_game_resolution()
        AgentServer.join()
        AgentServer.shut_down()
        logger.info("AgentServer关闭")
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.error("考虑重新配置环境")
        sys.exit(1)
    except Exception as e:
        logger.exception("agent运行过程中发生异常")
        raise


# -----
# region 程序入口
# -----


def main():
    current_version = read_interface_version()
    is_dev_mode = current_version == "DEBUG"

    _check_chinese_path()

    if sys.platform.startswith("win"):
        _check_admin_privilege()

    # 如果是Linux系统或开发模式，启动虚拟环境
    if sys.platform.startswith("linux") or is_dev_mode:
        ensure_venv_and_relaunch_if_needed()

    check_and_install_dependencies()

    if is_dev_mode:
        os.chdir(Path("./assets"))
        logger.info(f"set cwd: {os.getcwd()}")

    agent(is_dev_mode=is_dev_mode)


if __name__ == "__main__":
    main()