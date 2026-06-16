import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Callable


_VK = {
    "LButton": 0x01,
    "RButton": 0x02,
    "Q": 0x51,
    "E": 0x45,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
}


@dataclass(frozen=True)
class CombatConfig:
    character_keys: tuple[str, ...] = ("1", "2", "3", "4")
    skill_keys: tuple[str, ...] = ("E", "Q")
    final_character_key: str = "1"
    key_down_time: float = 0.05
    character_switch_delay: float = 0.35
    skill_cast_delay: float = 0.65
    basic_attack_duration: float = 10.0
    basic_attack_click_down_time: float = 0.03
    basic_attack_click_interval: float = 0.12
    tick_interval: float = 0.05
    round_count: int = 0

    @classmethod
    def from_mapping(cls, params: dict[str, Any] | None) -> "CombatConfig":
        if not params:
            return cls()

        config = cls()
        if "character_keys" in params:
            config = replace(
                config,
                character_keys=_normalize_keys(params["character_keys"]),
            )
        if "skill_keys" in params:
            config = replace(config, skill_keys=_normalize_keys(params["skill_keys"]))
        if "final_character_key" in params:
            config = replace(
                config,
                final_character_key=str(params["final_character_key"]).upper(),
            )

        numeric_fields = (
            "key_down_time",
            "character_switch_delay",
            "skill_cast_delay",
            "basic_attack_duration",
            "basic_attack_click_down_time",
            "basic_attack_click_interval",
            "tick_interval",
        )
        for field_name in numeric_fields:
            if field_name in params:
                config = replace(
                    config,
                    **{
                        field_name: _to_float(
                            params[field_name],
                            getattr(config, field_name),
                        )
                    },
                )
        if "round_count" in params:
            config = replace(
                config,
                round_count=_to_int(params["round_count"], config.round_count),
            )

        return config


class AutoCombat:
    """Combat loop: 1-2-3-4 cast E/Q, switch back to 1, click attack."""

    def __init__(
        self,
        controller: Any,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.controller = controller
        self.should_stop = should_stop or (lambda: False)

    def run_once(self, config: CombatConfig | None = None) -> bool:
        config = config or CombatConfig()

        for character_key in config.character_keys:
            if not self._press_key(character_key, config.key_down_time):
                return False
            if not self._sleep(config.character_switch_delay, config.tick_interval):
                return False

            for skill_key in config.skill_keys:
                if not self._press_key(skill_key, config.key_down_time):
                    return False
                if not self._sleep(config.skill_cast_delay, config.tick_interval):
                    return False

        if not self._press_key(config.final_character_key, config.key_down_time):
            return False
        if not self._sleep(config.character_switch_delay, config.tick_interval):
            return False

        return self._click_left_attack(
            duration=config.basic_attack_duration,
            click_down_time=config.basic_attack_click_down_time,
            click_interval=config.basic_attack_click_interval,
            tick_interval=config.tick_interval,
        )

    def run_loop(self, config: CombatConfig | None = None) -> bool:
        config = config or CombatConfig()
        rounds_done = 0

        while not self.should_stop():
            if config.round_count > 0 and rounds_done >= config.round_count:
                return True
            if not self.run_once(config):
                return bool(self.should_stop())
            rounds_done += 1

        return True

    def _press_key(self, key: str, hold_time: float) -> bool:
        vk = _key_to_vk(key)
        if vk is None or self.should_stop():
            return False

        return self._press_vk(vk, hold_time)

    def _press_vk(self, vk: int, hold_time: float) -> bool:
        if self.should_stop():
            return False

        self._wait_job(self.controller.post_key_down(vk))
        try:
            return self._sleep(hold_time, 0.01)
        finally:
            self._wait_job(self.controller.post_key_up(vk))

    def _click_left_attack(
        self,
        duration: float,
        click_down_time: float,
        click_interval: float,
        tick_interval: float,
    ) -> bool:
        if duration == 0 or self.should_stop():
            return not self.should_stop()

        click_down_time = max(0.001, click_down_time)
        click_interval = max(0.001, click_interval)
        tick_interval = max(0.001, tick_interval)
        deadline = None if duration < 0 else time.time() + duration

        while not self.should_stop():
            if deadline is not None and time.time() >= deadline:
                return True
            if not self._press_vk(_VK["LButton"], click_down_time):
                return False
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return True
                wait_time = min(click_interval, remaining)
            else:
                wait_time = click_interval
            if not self._sleep(wait_time, tick_interval):
                return False

        return False

    def _sleep(self, duration: float, tick_interval: float) -> bool:
        if duration <= 0:
            return not self.should_stop()

        tick_interval = max(0.001, tick_interval)
        deadline = time.time() + duration
        while time.time() < deadline:
            if self.should_stop():
                return False
            time.sleep(min(tick_interval, max(0.0, deadline - time.time())))
        return not self.should_stop()

    @staticmethod
    def _wait_job(job: Any) -> None:
        if hasattr(job, "wait"):
            job.wait()


def _normalize_keys(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        keys: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        keys = value
    else:
        return ()
    return tuple(str(key).upper() for key in keys if str(key).strip())


def _key_to_vk(key: str) -> int | None:
    key = str(key).upper()
    return _VK.get(key)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
