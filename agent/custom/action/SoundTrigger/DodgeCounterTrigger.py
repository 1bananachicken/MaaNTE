import random
import threading
import time
from typing import Callable, Optional

VK_SHIFT = 0xA0

logger = None


def _log():
    global logger
    if logger is None:
        from custom.action.Common.logger import get_logger

        logger = get_logger(__name__)
    return logger


class Dodger:
    def __init__(
        self,
        controller=None,
        dodge_fn: Optional[Callable] = None,
        counter_fn: Optional[Callable] = None,
        stop_check=None,
    ):
        self.controller = controller
        self.dodge_fn = dodge_fn or self._default_dodge
        self.counter_fn = counter_fn or self._default_counter
        self.stop_check = stop_check

        self._busy = False
        self._lock = threading.Lock()
        self._last_dodge = 0.0
        self._last_counter = 0.0
        self._dodge_cd = 0.5
        self._counter_cd = 1.0

    def dodge(self):
        if self.stop_check and self.stop_check():
            return
        if time.time() - self._last_dodge < self._dodge_cd:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.dodge_fn()
            self._last_dodge = time.time()
        except Exception as e:
            _log().error(f"Dodge failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def counter(self):
        if self.stop_check and self.stop_check():
            return
        if time.time() - self._last_counter < self._counter_cd:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.counter_fn()
            self._last_counter = time.time()
        except Exception as e:
            _log().error(f"Counter failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _click_key(self, key):
        if self.controller:
            self.controller.post_click_key(key)
        else:
            _log().warning("No controller, skip key click")

    def _default_dodge(self):
        _log().info("执行按键: 左Shift按下 -> 左Shift松开")
        self._click_key(VK_SHIFT)
        time.sleep(0.1 + random.random() * 0.2)
        self._click_key(VK_SHIFT)

    def _default_counter(self):
        key = random.choice([0x31, 0x32, 0x33, 0x34])
        key_name = chr(key)
        _log().info(
            f"执行按键: {key_name}按下 -> {key_name}松开 -> 左Shift按下 -> 左Shift松开"
        )
        self._click_key(key)
        time.sleep(0.02)
        self._click_key(VK_SHIFT)
