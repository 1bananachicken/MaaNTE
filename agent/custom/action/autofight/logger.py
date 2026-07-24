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

    _root_handler = None  # type: logging.Handler | None

    def setup_logger(log_dir="debug/custom", console_level="INFO"):
        """设置标准 logging logger

        Args:
            log_dir: 日志文件目录
            console_level: 控制台输出等级 (DEBUG, INFO, WARNING, ERROR)
        """
        global _root_handler

        os.makedirs(log_dir, exist_ok=True)

        _logger = logging.getLogger("autofight")
        _logger.setLevel(logging.DEBUG)

        # 移除旧的 handler（如果存在）
        if _root_handler is not None:
            _logger.removeHandler(_root_handler)

        # 控制台 handler
        _root_handler = logging.StreamHandler(sys.stderr)
        _root_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        _root_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        _logger.addHandler(_root_handler)

        # 文件 handler
        from datetime import datetime

        file_handler = logging.FileHandler(
            f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
            )
        )
        _logger.addHandler(file_handler)

        return _logger

    def change_console_level(level="DEBUG"):
        """动态修改控制台日志等级"""
        global _root_handler
        _logger = logging.getLogger("autofight")
        if _root_handler is not None:
            _root_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        else:
            _logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        _logger.info(f"控制台日志等级已更改为: {level}")

    logger = setup_logger()