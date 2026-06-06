import logging

from utils.logger import get_logger as _get_shared_logger


def get_logger(name: str) -> logging.Logger:
    return _get_shared_logger(name)
