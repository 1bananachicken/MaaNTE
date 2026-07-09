import os
import sys

try:
    from loguru import logger as _logger

    def setup_logger(log_dir="debug/custom", console_level="INFO"):
        """设置 loguru logger

        Args:
            log_dir: 日志文件目录
            console_level: 控制台输出等级 (DEBUG, INFO, WARNING, ERROR)
        """
        os.makedirs(log_dir, exist_ok=True)
        _logger.remove()

        _logger.add(
            sys.stderr,
            format="[<level>{level}</level>] <level>{message}</level>",
            colorize=True,
            level=console_level,
        )
        _logger.add(
            f"{log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation="00:00",  # midnight
            retention="2 weeks",
            compression="zip",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,  # 包含完整的异常回溯信息
            diagnose=True,  # 包含变量值信息
        )
        return _logger

    def change_console_level(level="DEBUG"):
        """动态修改控制台日志等级"""
        setup_logger(console_level=level)
        _logger.info(f"控制台日志等级已更改为: {level}")

    logger = setup_logger()
except ImportError:
    import logging

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    logger = logging