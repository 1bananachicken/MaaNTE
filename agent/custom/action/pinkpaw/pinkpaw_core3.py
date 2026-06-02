from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

try:
    from agent.custom.action.pinkpaw.pinkpaw_reward_logger import notify_pinkpaw_reward
except ImportError:
    from .pinkpaw_reward_logger import notify_pinkpaw_reward

VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
    "e": 0x45,
    "f": 0x46,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "esc": 0x1B,
    "lshift": 0xA0,
    "shift": 0x10,
}

MOUSE_VK = {
    "left": 0x01,
    "right": 0x02,
    "middle": 0x04,
}

REWARD_OCR_DELAY_MS = 3000
POST_REWARD_DELAY_MS = 7000
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_ROUTE_TIMING_SCALE = 1.0
MIN_ROUTE_TIMING_SCALE = 0.25
MAX_ROUTE_TIMING_SCALE = 1.2
MAX_ROUTE_SLEEP_ADJUST = 0.25
ROUTE_SLEEP_ADJUST_RATIO_CAP = 0.08
TIMING_SENSITIVE_KEYS = {"w", "a", "s", "d", "lshift", "space", "e"}


class AbortException(Exception):
    pass


class TaskerStoppedException(Exception):
    pass


@dataclass
class CharacterSwitchState:
    role: str
    keys: list[str]
    index: int = 0
    deadline: float = 0

    @property
    def current_key(self):
        return self.keys[self.index]

    def advance(self):
        self.index += 1
        return self.index < len(self.keys)


def _is_hit(result) -> bool:
    if result is None:
        return False
    status = getattr(result, "status", None)
    succeeded = getattr(status, "succeeded", None)
    if succeeded is not None:
        return bool(succeeded)
    if status is not None:
        return status == 0
    return bool(getattr(result, "hit", True))


def _norm_key(key: str) -> str:
    return str(key).lower()


def _parse_custom_action_param(argv: CustomAction.RunArg) -> dict:
    value = getattr(argv, "custom_action_param", None)
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception as exc:
        print(f"[PinkPawHeist/Core3] invalid custom_action_param: {value!r}, error: {exc}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_timing_scale(value) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = DEFAULT_ROUTE_TIMING_SCALE
    return max(MIN_ROUTE_TIMING_SCALE, min(MAX_ROUTE_TIMING_SCALE, scale))


class Core3ActionHelper:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mx, self.my = DEFAULT_WIDTH // 2, DEFAULT_HEIGHT // 2

    @property
    def controller(self):
        return getattr(getattr(self.ctx, "tasker", None), "controller", None)

    def is_stopping(self) -> bool:
        tasker = getattr(self.ctx, "tasker", None)
        if tasker is None:
            return False
        stopping = getattr(tasker, "stopping", False)
        if callable(stopping):
            stopping = stopping()
        return bool(stopping)

    def raise_if_stopped(self):
        if self.is_stopping():
            raise TaskerStoppedException("PinkPawHeistScheme3Action stopped by Maa tasker.")

    def run_task(self, task_name, pipeline_override=None):
        self.raise_if_stopped()
        if pipeline_override is None:
            result = self.ctx.run_task(task_name)
        else:
            result = self.ctx.run_task(task_name, pipeline_override=pipeline_override)
        self.raise_if_stopped()
        return result

    def _call_key(self, node_type, key_str, extra=None):
        if node_type != "KeyUp":
            self.raise_if_stopped()
        vk = VK.get(_norm_key(key_str))
        if vk is None:
            return False
        controller = self.controller
        if controller is not None:
            if node_type == "KeyDown":
                controller.post_key_down(vk)
            elif node_type == "KeyUp":
                controller.post_key_up(vk)
            elif node_type == "ClickKey":
                if hasattr(controller, "post_click_key"):
                    controller.post_click_key(vk)
                else:
                    controller.post_key_down(vk)
                    time.sleep(0.02)
                    controller.post_key_up(vk)
            if node_type != "KeyUp":
                self.raise_if_stopped()
            return True
        param = {"key": vk}
        if extra:
            param.update(extra)
        node_name = f"PinkPawHeist_{node_type}"
        override = {node_name: {"action": {"type": node_type, "param": param}}}
        ret = self.ctx.run_task(node_name, pipeline_override=override) is not None
        if node_type != "KeyUp":
            self.raise_if_stopped()
        return ret

    def click_key(self, key_str):
        return self._call_key("ClickKey", key_str)

    def key_down(self, key_str):
        return self._call_key("KeyDown", key_str)

    def key_up(self, key_str):
        return self._call_key("KeyUp", key_str)

    def move_to(self, x, y, duration_ms=None):
        self.raise_if_stopped()
        x, y = int(x), int(y)
        dx, dy = x - self.mx, y - self.my
        if dx * dx + dy * dy < 4:
            self.mx, self.my = x, y
            return True
        if duration_ms is None:
            duration_ms = max(int((dx**2 + dy**2) ** 0.5 / 0.5), 50)
        override = {
            "PinkPawHeist_MouseMove": {
                "action": {
                    "type": "Swipe",
                    "param": {
                        "begin": [self.mx, self.my],
                        "end": [x, y],
                        "duration": duration_ms,
                        "only_hover": True,
                    },
                }
            }
        }
        ret = self.ctx.run_task("PinkPawHeist_MouseMove", pipeline_override=override)
        self.raise_if_stopped()
        if ret:
            self.mx, self.my = x, y
        return ret

    def click(self, x, y):
        self.raise_if_stopped()
        controller = self.controller
        if controller is not None and hasattr(controller, "post_click"):
            controller.post_click(int(x), int(y))
            self.raise_if_stopped()
            self.mx, self.my = int(x), int(y)
            return True
        self.move_to(x, y)
        override = {
            "PinkPawHeist_Click": {
                "action": {"type": "Click", "param": {"target": [int(x), int(y)]}}
            }
        }
        ret = self.ctx.run_task("PinkPawHeist_Click", pipeline_override=override) is not None
        self.raise_if_stopped()
        return ret

    def mouse_down(self, key="left"):
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_down(vk)

    def mouse_up(self, key="left"):
        vk = MOUSE_VK.get(key, MOUSE_VK["left"])
        controller = self.controller
        if controller is not None:
            controller.post_key_up(vk)

    def release_controls(self):
        for key in ("w", "a", "s", "d", "e", "f", "space", "lshift"):
            try:
                self.key_up(key)
            except Exception as exc:
                print(f"[PinkPawHeist/Core3] failed to release {key}: {exc}")
        controller = self.controller
        if controller is None:
            return
        for vk in MOUSE_VK.values():
            try:
                controller.post_key_up(vk).wait()
            except Exception as exc:
                print(f"[PinkPawHeist/Core3] failed to release mouse {vk}: {exc}")


class PinkPawHeistCore3Path:
    CONF_FIGHTER = "fighter"
    CONF_RUNNER = "runner"
    CONF_AVOIDER = "avoider"
    CONF_AVOID_MTH = "avoid_method"
    ROLE_FIGHTER = "fighter"
    ROLE_RUNNER = "runner"
    ROLE_AVOIDER = "avoider"
    AVOID_METHOD_DASH = "dash"
    AVOID_METHOD_ATTACK = "attack"
    SWITCH_CHECK_DURATION = 1.0
    QUICK_PICK_START_DELAY = 0.3
    QUICK_PICK_INTERVAL = 0.2

    def __init__(self, ctx: Context, params: dict | None = None):
        self.ctx = ctx
        self.ah = Core3ActionHelper(ctx)
        self.exit_state = {1: False, 2: False, 3: False, 4: False}
        self.avoid_methods = [self.AVOID_METHOD_DASH, self.AVOID_METHOD_ATTACK]
        avoid_method = (params or {}).get(self.CONF_AVOID_MTH, self.AVOID_METHOD_DASH)
        if avoid_method not in self.avoid_methods:
            self.log_warning(f"unknown avoid_method {avoid_method!r}, fallback to dash")
            avoid_method = self.AVOID_METHOD_DASH
        self.route_timing_scale = _parse_timing_scale(
            (params or {}).get("timing_scale", DEFAULT_ROUTE_TIMING_SCALE)
        )
        self.config = {
            self.CONF_FIGHTER: ["4", "1"],
            self.CONF_RUNNER: ["3"],
            self.CONF_AVOIDER: ["2"],
            self.CONF_AVOID_MTH: avoid_method,
        }
        self._dead_fighter_keys: list[str] = []
        self._current_fighter_key: str | None = None
        self._switch_state: CharacterSwitchState | None = None
        self._held_keys: set[str] = set()
        self._quick_pick_active = False
        self._quick_pick_ready_at = 0.0
        self._next_quick_pick_at = 0.0
        self._last_action_at: dict[str, float] = {}
        self._interaction_watch_active = False
        self._interaction_watch_found = False
        self._checking_interaction = False
        self.last_check_reward_time = time.monotonic()
        self.check_reward_fail_count = 0
        self._round_label = "Core3"

    def log_info(self, *args):
        print("[PinkPawHeist/Core3]", *args)

    def log_warning(self, *args):
        print("[PinkPawHeist/Core3][WARN]", *args)

    def log_error(self, *args):
        print("[PinkPawHeist/Core3][ERROR]", *args)

    def log_round_info(self, message):
        self.log_info(f"{self._round_label}: {message}")
        self._log_to_frontend(str(message))

    def _log_to_frontend(self, message: str):
        try:
            self.ctx.run_task(
                "PinkPawHeist_LogMessage",
                pipeline_override={
                    "PinkPawHeist_LogMessage": {
                        "focus": {
                            "Node.Action.Starting": {
                                "content": f"[Core3] {message}",
                                "display": ["log", "toast"],
                            }
                        }
                    }
                },
            )
        except Exception:
            pass

    def _check_interval(self, name: str, interval: float) -> bool:
        if interval is None or interval < 0:
            return True
        now = time.monotonic()
        last = self._last_action_at.get(name, 0.0)
        if now - last < interval:
            return False
        self._last_action_at[name] = now
        return True

    def _poll_quick_pick(self):
        if not self._quick_pick_active:
            return
        now = time.monotonic()
        if now < self._quick_pick_ready_at or now < self._next_quick_pick_at:
            return
        self.ah.click_key("f")
        self._next_quick_pick_at = now + self.QUICK_PICK_INTERVAL

    def _has_timing_sensitive_key_held(self) -> bool:
        return bool(self._held_keys & TIMING_SENSITIVE_KEYS)

    def _check_still_in_heist(self):
        now = time.monotonic()
        if now - self.last_check_reward_time <= 2.0:
            return
        self.last_check_reward_time = now

        result = self.ah.run_task(
            "PinkPawHeist_CheckReward",
            pipeline_override={"PinkPawHeist_CheckReward": {"timeout": 100}},
        )
        if not _is_hit(result):
            self.check_reward_fail_count += 1
            self.log_warning(
                f"未检测到本局收益，连续失败 {self.check_reward_fail_count} 次"
            )
            if self.check_reward_fail_count >= 2:
                raise AbortException("PinkPawHeist_CheckReward 连续 2 次检测失败")
        else:
            self.check_reward_fail_count = 0

    def _scale_route_duration(self, duration: float) -> float:
        if duration <= 0 or self.route_timing_scale == 1.0:
            return max(duration, 0.0)

        wanted_adjust = duration * abs(1.0 - self.route_timing_scale)
        adaptive_cap = min(
            MAX_ROUTE_SLEEP_ADJUST,
            max(0.02, duration * ROUTE_SLEEP_ADJUST_RATIO_CAP),
        )
        adjust = min(wanted_adjust, adaptive_cap)
        if self.route_timing_scale < 1.0:
            return max(0.0, duration - adjust)
        return duration + adjust

    def sleep(self, timeout, check_reward=True, scaled=True):
        duration = max(float(timeout), 0.0)
        if scaled:
            duration = self._scale_route_duration(duration)
        target = time.monotonic() + duration
        while time.monotonic() < target:
            self.ah.raise_if_stopped()
            self._poll_quick_pick()
            timing_sensitive = self._has_timing_sensitive_key_held()
            if check_reward and not timing_sensitive:
                self._check_still_in_heist()
            if (
                self._interaction_watch_active
                and not self._interaction_watch_found
                and not self._checking_interaction
                and not timing_sensitive
            ):
                self._interaction_watch_found = self.find_interac()
            remaining = target - time.monotonic()
            time.sleep(max(0.0, min(0.05, remaining)))
        self._poll_quick_pick()
        return True

    def next_frame(self):
        self.sleep(0.05)
        return True

    def send_key(self, key, down_time=0.02, interval=-1, after_sleep=0, action_name=None):
        key = _norm_key(key)
        name = action_name or f"key:{key}"
        if not self._check_interval(name, interval):
            return False
        if key == "f":
            self.ah.click_key(key)
            if down_time and down_time > 0.06:
                self.sleep(down_time)
            if after_sleep:
                self.sleep(after_sleep)
            return True
        if down_time and down_time > 0.06:
            self.send_key_down(key)
            self.sleep(down_time)
            self.send_key_up(key)
        else:
            self.ah.click_key(key)
        if after_sleep:
            self.sleep(after_sleep)
        return True

    def send_key_down(self, key, after_sleep=0):
        key = _norm_key(key)
        if key == "f":
            if not self._quick_pick_active:
                self._quick_pick_ready_at = time.monotonic() + self.QUICK_PICK_START_DELAY
                self._next_quick_pick_at = self._quick_pick_ready_at
            self._quick_pick_active = True
            return True
        self._held_keys.add(key)
        ret = self.ah.key_down(key)
        if after_sleep:
            self.sleep(after_sleep)
        return ret

    def send_key_up(self, key, after_sleep=0):
        key = _norm_key(key)
        if key == "f":
            self._quick_pick_active = False
            return True
        try:
            return self.ah.key_up(key)
        finally:
            self._held_keys.discard(key)
            if after_sleep:
                self.sleep(after_sleep)

    def sleep_send_key(self, time_out, key, interval=0.2):
        deadline = time.monotonic() + time_out
        while time.monotonic() < deadline:
            self.send_key(key, interval=interval)
            self.sleep(0.01)

    def mouse_down(self, x=-1, y=-1, name=None, key="left"):
        self.ah.mouse_down(key=key)

    def mouse_up(self, name=None, key="left"):
        self.ah.mouse_up(key=key)

    def click(
        self,
        x=-1,
        y=-1,
        move_back=False,
        name=None,
        interval=-1,
        move=True,
        key="left",
        down_time=0.01,
        after_sleep=0,
    ):
        name = name or f"click:{key}"
        if not self._check_interval(name, interval):
            return False
        if x == -1:
            x = 0.5
        if y == -1:
            y = 0.5
        px = int(x * DEFAULT_WIDTH) if isinstance(x, float) and x <= 1 else int(x)
        py = int(y * DEFAULT_HEIGHT) if isinstance(y, float) and y <= 1 else int(y)
        if key == "left" and down_time <= 0.05:
            ret = self.ah.click(px, py)
        else:
            self.ah.move_to(px, py)
            self.ah.mouse_down(key=key)
            self.sleep(max(down_time, 0.01))
            self.ah.mouse_up(key=key)
            ret = True
        if after_sleep:
            self.sleep(after_sleep)
        return ret

    def wait_until(
        self,
        condition,
        time_out=0,
        pre_action=None,
        post_action=None,
        settle_time=-1,
        raise_if_not_found=False,
        **kwargs,
    ):
        timeout = 10.0 if not time_out or time_out <= 0 else float(time_out)
        deadline = time.monotonic() + timeout
        settled_at = None
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            if pre_action is not None:
                pre_action()
            found = bool(condition())
            if found:
                if post_action is not None:
                    post_action()
                if settle_time is not None and settle_time >= 0:
                    if settled_at is None:
                        settled_at = time.monotonic()
                    if time.monotonic() - settled_at >= settle_time:
                        return True
                else:
                    return True
            else:
                settled_at = None
            self.sleep(0.1)
        if raise_if_not_found:
            raise AbortException("timeout for wait_until")
        return False

    def wait_team_ui_settle(self):
        self.sleep(0.5)
        return True

    def _run_check_node(self, node_name, timeout=1.5):
        deadline = time.monotonic() + timeout
        override = {node_name: {"timeout": max(20, int(timeout * 1000))}}
        while time.monotonic() < deadline:
            self.ah.raise_if_stopped()
            if _is_hit(self.ah.run_task(node_name, pipeline_override=override)):
                return True
            time.sleep(0.05)
        return False

    def find_interac(self):
        self._checking_interaction = True
        try:
            if self._run_check_node("PinkPawHeist_Core3_CheckInteractPinkOnce", timeout=0.08):
                return True
            if self._run_check_node("PinkPawHeist_Core3_CheckInteractTemplateOnce", timeout=0.08):
                return True
            if self._run_check_node("PinkPawHeist_Core3_CheckInteractOnce", timeout=0.08):
                return True
            return any(
                self._run_check_node(node, timeout=0.04)
                for node in (
                    "PinkPawHeist_CheckDoorOnce",
                    "PinkPawHeist_CheckGateOnce",
                    "PinkPawHeist_CheckGate2Once",
                    "PinkPawHeist_CheckEvacuateOnce",
                )
            )
        finally:
            self._checking_interaction = False

    def wait_ocr(
        self,
        x=0,
        y=0,
        to_x=1,
        to_y=1,
        width=0,
        height=0,
        name=None,
        box=None,
        match=None,
        threshold=0,
        frame=None,
        target_height=0,
        time_out=0,
        post_action=None,
        raise_if_not_found=False,
        log=False,
        screenshot=False,
        settle_time=-1,
        lib="default",
    ):
        timeout = 1.5 if not time_out or time_out <= 0 else float(time_out)
        matched = self._run_check_node("PinkPawHeist_CheckDoorOnce", timeout=timeout)
        if not matched:
            matched = self._run_check_node("PinkPawHeist_Core3_CheckInteractPinkOnce", timeout=0.2)
        if not matched:
            matched = self._run_check_node("PinkPawHeist_Core3_CheckInteractTemplateOnce", timeout=0.2)
        if not matched:
            matched = self._run_check_node("PinkPawHeist_Core3_CheckInteractOnce", timeout=0.2)
        if matched:
            if post_action:
                post_action()
            return [type("OCRText", (), {"name": getattr(match, "pattern", "")})()]
        if raise_if_not_found:
            raise AbortException("timeout for wait_ocr")
        return None

    def start_interaction_watch(self):
        self._interaction_watch_active = True
        self._interaction_watch_found = False
        return True

    def stop_interaction_watch(self):
        self._interaction_watch_active = False
        self._interaction_watch_found = False
        return True

    def is_lock_pick_active(self):
        return self._run_check_node("PinkPawHeist_Core3_CheckLockPickActiveOnce", timeout=0.08) or self._quick_pick_active

    def wait_and_interact(
        self, direction=None, interact=True, key_up_sleep=0.7, is_lock=False, time_out=10
    ):
        ret = self.wait_until(self.find_interac, time_out=time_out)
        if interact and direction is not None:
            self.send_key_up(direction)
            self.sleep(key_up_sleep)
        if not ret:
            raise AbortException("timeout for wait_and_interact")
        if not interact:
            return True
        self.send_key("f", interval=-1)
        self.wait_until(
            lambda: not self.find_interac(),
            pre_action=lambda: self.send_key("f", interval=0.6, action_name="wait_and_interact_f"),
            time_out=2.0,
        )
        if is_lock:
            if self.wait_until(self.is_lock_pick_active, time_out=2.0):
                self.wait_until(
                    lambda: not self.is_lock_pick_active(),
                    time_out=max(float(time_out), 5.0),
                    settle_time=0.5,
                )
            else:
                self.sleep(max(float(time_out), 5.0), scaled=False)
        return True

    def loot_safes_while_walking(
        self, direction=None, min_walk_time=0, time_out=10, hold=False, send_pick=False
    ):
        start_time = time.monotonic()
        deadline = start_time + time_out
        if direction is not None:
            self.send_key_down(direction)
        pick_started = False
        while time.monotonic() < deadline:
            if send_pick and not pick_started and time.monotonic() - start_time >= min_walk_time:
                self.send_key_down("f")
                pick_started = True
            self._poll_quick_pick()
            self.sleep(0.05)
        if direction is not None and not hold:
            self.send_key_up(direction)
        if send_pick and pick_started:
            self.send_key_up("f")

    def wait_for_safe_loot(self, time_out=10, raise_timeout=False):
        self.sleep(min(float(time_out), 1.2))
        return True

    def has_extract_panel(self):
        return self._run_check_node("PinkPawHeist_CheckEvacuateOnce", timeout=0.2)

    def try_open_exit(self, direction=None):
        if not self.wait_until(self.find_interac, time_out=4):
            raise AbortException("not found exit interaction")
        if direction is not None:
            self.send_key_up(direction)
            self.sleep(0.3)
        ret = self.wait_until(
            self.has_extract_panel,
            pre_action=lambda: self.send_key("f", interval=1),
            time_out=1.75,
        )
        if ret:
            self.sleep(0.3)
            self.send_key("esc", interval=0.5)
            self.sleep(0.5)
        return ret

    def walk_until_extract_panel(self, direction=None, time_out=10):
        if direction is not None:
            self.send_key_down(direction)
        try:
            return self.wait_until(
                self.has_extract_panel,
                pre_action=lambda: self.send_key("f", interval=0.25),
                time_out=time_out,
                raise_if_not_found=True,
            )
        finally:
            if direction is not None:
                self.send_key_up(direction)

    def clear_current_combat(self):
        self.switch_to_fighter(check_switched=True)
        self.fight_until_no_monster(timeout_no_monster=10000, wait_for_monster=True)
        self.switch_to_runner(check_switched=True)

    def check_monster(self):
        image = self.ctx.tasker.controller.post_screencap().wait().get()
        result = self.ctx.run_recognition("PinkPawHeist_CheckMonsterOnce", image)
        return result is not None and getattr(result, "hit", False)

    def wait_monster(self, timeout=6000):
        deadline = time.monotonic() + timeout / 1000.0
        while time.monotonic() < deadline:
            if self.check_monster():
                return True
            self.sleep(0.2)
        return False

    def attack_cycle(self, times=3, loot=False):
        for _ in range(times):
            self.ah.run_task("PinkPawHeist_Core1_Attack_Space")
        if loot:
            self.send_key("f")

    def fight_until_no_monster(
        self,
        timeout_no_monster=10000,
        wait_for_monster=True,
        role_to_switch_back=None,
        loot=False,
        attack_cycles=3,
    ):
        if wait_for_monster and not self.wait_monster(timeout=timeout_no_monster):
            return False
        no_monster_start = None
        while True:
            if self.check_monster():
                no_monster_start = None
                self.attack_cycle(times=attack_cycles, loot=loot)
            else:
                now = time.monotonic()
                if no_monster_start is None:
                    no_monster_start = now
                elif now - no_monster_start >= timeout_no_monster / 1000.0:
                    break
                self.sleep(0.05)
        if role_to_switch_back:
            self.switch_to_key(role_to_switch_back)
        return True

    def switch_to_key(self, key):
        for _ in range(3):
            self.send_key(str(key))
            self.sleep(0.2)
        return str(key)

    def _begin_character_switch(self, role, keys, check_switched=False):
        keys = [str(key) for key in keys]
        if not keys:
            raise AbortException(f"{role} {keys} dead or empty")
        key = keys[0]
        if role == self.ROLE_FIGHTER:
            self._current_fighter_key = key
        return self.switch_to_key(key)

    def switch_to_runner(self, check_switched=False):
        return self._begin_character_switch(self.ROLE_RUNNER, self.config.get(self.CONF_RUNNER, []), check_switched)

    def switch_to_avoider(self, check_switched=False):
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            self.log_info("no avoider")
            return None
        return self._begin_character_switch(self.ROLE_AVOIDER, keys, check_switched)

    def avoider_strategy_index(self):
        keys = self.config.get(self.CONF_AVOIDER, [])
        if not keys:
            return -1
        method_name = self.config.get(self.CONF_AVOID_MTH)
        if method_name not in self.avoid_methods:
            return 0
        return self.avoid_methods.index(method_name)

    def perform_avoidance_action(self):
        method_name = self.config.get(self.CONF_AVOID_MTH)
        if method_name == self.AVOID_METHOD_ATTACK:
            self.click(down_time=0.6)
            return
        self.send_key_down("w")
        self.sleep(0.1)
        self.send_key_down("lshift")
        self.sleep(1.0)
        self.send_key_up("lshift")
        self.sleep(0.1)
        self.send_key_up("w")

    def exit_heist(self):
        self.log_round_info("Confirm extract")
        self.sleep(1.0, check_reward=False, scaled=False)
        result = self.ah.run_task("PinkPawHeist_EvacuateOnce")
        if _is_hit(result):
            self.sleep(REWARD_OCR_DELAY_MS / 1000.0, check_reward=False, scaled=False)
            notify_pinkpaw_reward(self.ctx, success=True)
            self.sleep(POST_REWARD_DELAY_MS / 1000.0, check_reward=False, scaled=False)
            return True
        notify_pinkpaw_reward(self.ctx, success=False)
        return False

    def abort_heist(self):
        self.log_round_info("Abort and return to main")
        self.ah.release_controls()
        for _ in range(4):
            self.send_key("esc")
            self.sleep(1.0, check_reward=False, scaled=False)
        self.ah.run_task("PinkPawHeist_Once")
        self.sleep(5.0, check_reward=False, scaled=False)
        notify_pinkpaw_reward(self.ctx, success=False)

    def _release_held_keys(self):
        held = list(self._held_keys)
        self._held_keys.clear()
        for key in held:
            try:
                self.ah.key_up(key)
            except Exception as exc:
                self.log_error(f"release held key {key} failed", exc)
        self._quick_pick_active = False

    def run_path(self):
        self.goto_lg1()
        self.wait_team_ui_settle()
        # self.check_current_floor(1)
        self.lg1_wp1()
        self.lg1_wp2()
        self.lg1_wp3()
        self.lg1_wp4()
        idx = self.avoider_strategy_index()
        if idx == -1:
            self.lg1_wp5_avoid_combat_01()
        elif idx == 0:
            self.lg1_wp5_avoid_combat_02()
        elif idx == 1:
            self.lg1_wp5_avoid_combat_03()
        self.wait_team_ui_settle()
        # self.check_current_floor(2)
        self.lg2_wp1_to_exit1()
        self.lg2_wp1_remains()
        self.lg2_wp2_to_exit2()
        self.lg2_wp3_to_layzer_room()
        self.lg2_wp3_in_layzer_room()
        self.lg2_wp4()
        if self.exit_state[1]:
            self.lg2_wp4_to_exit1()
        elif self.exit_state[2]:
            self.lg2_wp4_to_exit2()
        else:
            self.lg2_wp4_to_exit3()

    def goto_lg1(self):
        self.log_round_info("寻路到LG1")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.81)
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key_down("lshift")
        self.sleep(0.16)
        self.send_key_up("lshift")
        self.sleep(2.68)
        self.send_key_down("d")
        self.sleep(2.55)
        self.send_key_up("d")
        self.sleep(0.37)
        self.wait_and_interact(direction="w", is_lock=True)
        self.send_key_down("w")
        self.sleep(0.25)

        self.send_key_down("f")
        start = time.time()
        while time.time() < start + 10:
            self.send_key("space", down_time=0.14, interval=0.25)
            if time.time() > start + 6.4 and self.find_interac():
                break
            self.next_frame()

        self.wait_until(self.is_lock_pick_active, settle_time=0.5)
        self.send_key_up("f")
        self.send_key_up("w")
        self.wait_until(lambda: not self.is_lock_pick_active(), settle_time=0.5)
        if self.find_interac():
            self.goto_lg1_interrupted()
        self.sleep(0.01)

        self.send_key_down("w")
        self.sleep(0.2)
        self.sleep_send_key(0.2, key="lshift")
        self.send_key_down("d")
        self.sleep_send_key(0.5, key="lshift")
        self.sleep(0.5)
        self.send_key_up("d")
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep_send_key(0.5, key="lshift")
        self.sleep(0.5)
        self.send_key_up("w")
        self.sleep_send_key(3.5, interval=0.7, key="lshift")
        self.send_key_up("a")

        self.sleep(0.04)
        self.send_key_down("s")
        self.sleep(0.29)
        self.send_key("lshift")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.04)
        self.send_key_down("d")
        self.sleep(0.29)
        self.send_key("lshift")
        self.sleep(2.50)
        self.send_key_up("d")
        self.sleep(0.40)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.36)
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.14)
        self.send_key_down("f")  # start pick
        self.sleep(0.04)
        self.send_key_down("w")
        self.sleep(2.5)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_up("f")  # end pick
        self.sleep(0.13)
        self.send_key_down("s")
        self.sleep(0.14)
        self.send_key_up("s")
        self.sleep(0.20)
        self.clear_current_combat()
        self.send_key_down("f")
        self.sleep(0.5)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("lshift")
        self.sleep(0.10)
        self.send_key_up("lshift")
        self.sleep(1)
        self.send_key_up("f")
        self.sleep(0.1)
        self.send_key_down("d")
        self.sleep(1)
        self.send_key_up("w")
        self.sleep(1.5)
        self.send_key_up("d")
        self.sleep(0.35)
        self.send_key_down("a")
        self.sleep(0.88)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("s")
        self.sleep(0.43)
        self.send_key_down("lshift")
        self.sleep(0.13)
        self.send_key_up("lshift")
        self.sleep(1.4)
        self.send_key_down("a")
        self.sleep(0.53)
        self.send_key_up("a")
        self.sleep(1.64)
        self.wait_and_interact(direction="s")
        self.sleep(0.50)
        self.send_key_down("s")
        self.sleep(0.10)
        self.wait_and_interact(direction="s")

    def goto_lg1_interrupted(self):
        self.log_round_info("LG1开锁中断恢复")
        self.clear_current_combat()
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(3.41)
        self.send_key_up("d")
        self.sleep(0.32)
        self.send_key_down("a")
        self.sleep(1.16)
        self.send_key_up("a")
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(1.51)
        self.send_key_up("w")
        self.sleep(0.11)
        self.wait_and_interact(direction="w", is_lock=True)
        self.sleep(0.5)

    def lg1_wp1(self):
        self.log_round_info("LG1 WP1")
        self.sleep(0.75)
        self.send_key_down("w")
        self.sleep(9.06)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.71)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.01)
        self.wait_and_interact(direction="s", key_up_sleep=0)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.25)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.03)
        self.send_key_up("d")
        self.sleep(0.22)
        self.send_key_down("a")
        self.sleep(3.90)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(0.40)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.60)
        self.send_key_up("d")
        self.sleep(0.06)
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.21)
        self.send_key_up("d")
        self.sleep(0.12)

    def lg1_wp2(self):
        self.log_round_info("LG1 WP2")
        self.send_key_down("d")
        self.sleep(1.80)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.71)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.28)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.26)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(2.93)
        self.send_key_down("w")  # 过镭射1
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.44)
        self.start_interaction_watch()
        self.send_key_down("w")  # 过镭射2
        self.sleep(8.51)
        self.send_key_up("w")
        self.stop_interaction_watch()
        self.sleep(0.33)

    def lg1_wp3(self):
        self.log_round_info("LG1 WP3")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(2.13)
        self.send_key_up("a")
        self.sleep(0.52)
        self.send_key_down("s")
        self.sleep(1.32)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.20)
        self.send_key_up("w")
        self.sleep(0.31)
        self.send_key_down("a")
        self.sleep(1.50)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key_up("d")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("w")
        self.sleep(3.19)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(5.16)
        self.send_key_up("w")
        self.stop_interaction_watch()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.15)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(5.31)
        self.send_key_up("w")
        self.sleep(0.12)

    def lg1_wp4(self):
        self.log_round_info("LG1 WP4")
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.31)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.50)
        self.send_key_up("w")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_up("a")
        self.sleep(1.22)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(6.58)
        self.send_key_down("d")
        self.sleep(2.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(1.30)
        self.start_interaction_watch()
        self.send_key_down("d")
        self.sleep(2.10)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.65)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.48)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.14)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(1.41)
        self.send_key_down("w")
        self.sleep(0.81)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.47)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.11)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(3.38)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.loot_safes_while_walking(direction="s", time_out=2.37)
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(1.33)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(9.40)
        self.send_key_up("w")
        self.sleep(0.31)

    def lg1_wp5_avoid_combat_01(self):
        self.log_round_info("LG1 WP5避战路线1")
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(0.81)
        self.send_key_up("s")
        self.sleep(2.01)
        self.send_key_down("w")
        self.sleep(0.11)

        deadline = time.time() + 4.5
        while time.time() < deadline:
            self.send_key("lshift")
            self.sleep(0.51)

        self.wait_and_interact(direction="w", is_lock=True)
        self.sleep(0.11)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg1_wp5_avoid_combat_02(self):
        self.log_round_info("LG1 WP5避战路线2")
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_up("s")
        self.sleep(0.11)

        self.switch_to_avoider(check_switched=True)
        self.sleep(0.5)
        self.perform_avoidance_action()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(6.0)
        self.send_key_up("w")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.5)
        self.wait_and_interact(is_lock=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg1_wp5_avoid_combat_03(self):
        self.log_round_info("LG1 WP5避战路线3")
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.5)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.send_key_down("w")
        self.sleep(0.11)

        deadline = time.time() + 4.5
        while time.time() < deadline:
            self.send_key("lshift")
            self.sleep(0.51)

        self.send_key_up("w")
        self.sleep(0.2)
        self.perform_avoidance_action()
        self.sleep(3.2)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.5)
        self.wait_and_interact(is_lock=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.00)
        self.wait_and_interact(direction="w")

    def lg2_wp1_to_exit1(self):
        self.log_round_info("LG2 WP1尝试出口1")
        self.sleep(2.65)  # 2.65
        self.send_key_down("w")
        self.sleep(5.04)
        self.send_key_up("w")
        self.sleep(0.13)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(3.00)
        self.send_key("lshift")  # x0.6
        self.sleep(3.10)
        self.send_key_up("a")
        self.sleep(0.21)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("s")
        self.sleep(0.33)
        self.send_key_down("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.41)
        self.send_key_up("d")
        self.sleep(1.21)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.52)
        self.send_key_up("d")
        self.sleep(0.59)
        self.send_key_down("s")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.89)
        self.send_key_up("d")
        self.sleep(0.41)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.81)
        self.exit_state[1] = self.try_open_exit(direction="w")

    def lg2_wp1_remains(self):
        self.log_round_info("LG2 WP1剩余路线")
        self.send_key_down("w")
        self.sleep(2.10)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.90)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("w")
        self.sleep(0.80)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.06)
        self.send_key_up("w")
        self.sleep(0.11)
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.81)
        self.send_key_up("a")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.2)
        self.send_key("lshift")
        self.sleep(1.43)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.01)
        self.send_key_up("w")
        self.sleep(0.18)
        self.send_key_down("d")
        self.sleep(0.70)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("s")
        self.sleep(3.02)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.72)
        self.send_key_down("s")
        self.sleep(6.36)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.05)
        self.send_key_up("d")
        self.sleep(0.23)
        self.switch_to_runner()
        self.sleep(0.01)
        self.send_key_up("f")  # end pick
        self.loot_safes_while_walking(
            direction="w", min_walk_time=0.8, time_out=1.3, hold=True, send_pick=True
        )
        self.sleep(0.10)
        self.send_key_down("space")
        self.sleep(0.13)
        self.send_key_up("space")
        self.sleep(0.17)
        self.send_key_down("space")
        self.sleep(0.13)
        self.send_key_up("space")
        self.sleep(7.46)
        self.send_key_down("d")
        self.sleep(1.31)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.14)
        self.send_key_down("s")
        self.sleep(0.22)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.71)
        self.send_key_up("w")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.72)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.90)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.02)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.30)
        self.send_key_down("a")
        self.sleep(0.60)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.26)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.01)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.29)
        self.send_key_down("a")
        self.sleep(0.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(7.12)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(1.44)
        self.send_key_up("d")
        self.sleep(0.91)
        self.send_key_up("w")
        self.sleep(0.30)
        self.send_key_down("d")
        self.sleep(0.72)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.61)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)

    def lg2_wp2_to_exit2(self):
        self.log_round_info("LG2 WP2尝试出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.21)
        self.send_key_down("space")
        self.sleep(0.06)
        self.send_key_up("space")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(1.70)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.20)
        self.send_key("lshift")
        self.sleep(2.64)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.96)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.15)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.61)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.20)
        self.send_key_down("a")
        self.sleep(0.72)
        self.send_key_up("a")
        self.sleep(1.26)
        self.send_key_down("w")
        self.sleep(2.60)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.31)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.63)  # 4.03
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.75)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.56)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(1.57)
        self.exit_state[2] = self.try_open_exit(direction="w")
        self.sleep(0.40)

    def lg2_wp3_to_layzer_room(self):
        self.log_round_info("LG2 WP3前往镭射房")
        self.send_key_down("a")
        self.sleep(3.03)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.55)
        self.send_key_up("w")
        self.sleep(0.51)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.56)
        self.send_key_up("s")
        self.sleep(1.18)
        self.send_key_down("a")
        self.sleep(2.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.77)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.60)
        self.send_key_up("a")
        self.sleep(0.29)
        self.send_key_down("d")
        self.sleep(1.31)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.76)
        self.send_key_up("s")
        self.sleep(0.30)
        self.send_key_down("a")
        self.sleep(0.61)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.96)
        self.send_key_up("s")

    def lg2_wp3_in_layzer_room(self):
        self.log_round_info("LG2 WP3镭射房")
        self.send_key_down("d")
        self.sleep(0.36)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.wait_for_safe_loot(time_out=0.8)
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.46)
        self.send_key_up("a")

        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.40)
        self.send_key_up("w")
        self.sleep(0.36)
        self.send_key_down("a")
        self.sleep(0.51)
        self.send_key_down("w")
        self.sleep(1.00)
        self.send_key_up("a")
        self.send_key_up("w")

        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.30)
        self.send_key_down("s")
        self.sleep(0.30)
        self.send_key_up("s")

        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.41)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.30)
        self.send_key_up("d")
        self.sleep(0.54)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.16)
        self.send_key_down("a")
        self.sleep(0.33)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.33)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.40)
        self.send_key_down("s")
        self.sleep(0.20)
        self.send_key_down("space")
        self.sleep(0.07)
        self.send_key_up("space")
        self.sleep(1.21)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.31)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.74)
        self.send_key_up("a")
        self.sleep(0.80)
        self.send_key_down("d")
        self.sleep(0.71)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.52)
        self.send_key_up("s")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("f")  # start pick
        self.wait_for_safe_loot(raise_timeout=True)
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.90)
        self.send_key_up("d")
        self.sleep(1.32)
        self.send_key_up("w")
        self.sleep(0.92)
        self.send_key_down("s")
        self.sleep(0.08)
        self.send_key_down("d")
        self.sleep(1.36)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.51)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.17)
        self.send_key_up("s")
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(2.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.63)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.51)
        self.send_key_up("a")
        self.sleep(0.13)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.39)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_down("a")
        self.sleep(3.01)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.51)
        self.send_key_up("w")
        self.sleep(0.13)
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.31)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.73)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.43)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_down("d")
        self.sleep(0.51)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.22)
        self.send_key_down("s")
        self.sleep(0.03)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.51)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(1.21)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.31)
        self.send_key_down("s")
        self.sleep(0.40)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_down("a")
        self.sleep(1.41)
        self.send_key_up("a")
        self.sleep(0.11)

    def lg2_wp4(self):
        self.log_round_info("LG2 WP4")
        self.send_key_down("w")
        self.sleep(4.40)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.20)
        self.send_key_down("lshift")
        self.sleep(0.20)
        self.send_key_up("lshift")
        self.sleep(1.5)
        self.send_key_up("a")
        self.sleep(0.01)
        self.send_key_up("f")  # end pick

    def lg2_wp4_to_exit1(self):
        self.log_round_info("LG2 WP4前往出口1")
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("a")
        self.sleep(0.17)
        self.send_key_down("lshift")
        self.sleep(0.14)
        self.send_key_up("lshift")
        self.sleep(4.69)
        self.send_key_up("a")
        self.sleep(0.41)
        self.send_key_down("d")
        self.sleep(0.31)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_down("s")
        self.sleep(1.50)
        self.send_key_down("lshift")
        self.sleep(0.23)
        self.send_key_up("lshift")
        self.sleep(4.55)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)

        deadline = time.time() + 1.29
        while time.time() < deadline:
            self.send_key("space")
            self.sleep(0.25)

        self.sleep(1.21)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(0.11)
        self.walk_until_extract_panel(direction="w")

    def lg2_wp4_to_exit2(self):
        self.log_round_info("LG2 WP4前往出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.01)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(0.11)
        self.send_key_down("lshift")
        self.sleep(0.06)
        self.send_key_up("lshift")
        self.sleep(1.10)
        self.send_key_down("d")
        self.sleep(0.90)
        self.send_key_up("w")
        self.sleep(2.30)
        self.send_key_down("s")
        self.sleep(1.01)
        self.send_key_up("s")
        self.sleep(0.21)
        self.send_key_down("w")
        self.sleep(0.74)
        self.send_key_up("w")
        self.sleep(4.61)
        self.send_key_up("d")
        self.sleep(0.41)
        self.send_key_down("s")
        self.sleep(1.00)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.00)
        self.send_key_down("w")
        self.sleep(1.00)
        self.send_key_up("w")
        self.walk_until_extract_panel(direction="d")

    def lg2_wp4_to_exit3(self):
        self.log_round_info("LG2 WP4前往出口3")
        self.send_key_down("w")
        self.sleep(0.14)
        self.send_key_down("lshift")
        self.sleep(0.13)
        self.send_key_up("lshift")
        self.sleep(2.70)
        self.send_key_down("a")
        self.sleep(1.98)
        self.send_key_up("a")
        self.wait_and_interact(direction="w", is_lock=True, time_out=6)
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.05)
        self.send_key_down("lshift")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.05)
        self.send_key_up("lshift")
        self.sleep(1.76)
        self.send_key_up("w")
        self.sleep(0.26)
        self.send_key_up("d")
        self.sleep(0.10)
        self.walk_until_extract_panel(direction="d")

    def run_path(self):
        idx = self.avoider_strategy_index()
        if idx == -1:
            self.log_round_info("没有配置避战角色，全程使用原始线路（路线A）")
            self.goto_lg1()
        elif idx == 0:
            self.log_round_info("配置避战角色狗哥，使用早雾避战（路线B）")
            self.goto_lg1_skip_Sakiri()
        elif idx == 1:
            self.log_round_info("配置避战角色浔，使用浔避战（路线B）")
            self.goto_lg1_skip_Hotori()
        self.wait_team_ui_settle()
        # if not self.check_current_floor_str("办公"):
        #     self.check_current_floor(1)
        self.switch_to_runner(check_switched=True)
        self.lg1_wp1_safer()
        self.lg1_wp2()
        self.lg1_wp3()
        if idx == -1:
            self.lg1_wp4()
            self.lg1_wp5_avoid_combat_01()
        elif idx == 0:
            self.lg1_wp4_buster()
            self.lg1_wp5_buster()
        elif idx == 1:
            self.lg1_wp4()
            self.lg1_wp5_avoid_combat_03()
        self.wait_team_ui_settle()
        # if not self.check_current_floor_str("藏品"):
        #     self.check_current_floor(2)
        self.lg2_wp1_to_exit1() # self.lg2_wp1_to_exit1_safer(False)
        self.lg2_wp1_remains()
        self.lg2_wp2_to_exit2_safer()
        self.lg2_wp3_to_layzer_room()
        self.lg2_wp3_in_layzer_room()
        self.lg2_wp4()
        if self.exit_state[1]:
            self.lg2_wp4_to_exit1()
        elif self.exit_state[2]:
            self.lg2_wp4_to_exit2()
        else:
            self.lg2_wp4_to_exit3()

    def goto_lg1_skip_Sakiri(self):
        self.log_round_info("早雾、大厅前往LG1")
        self.sleep(0.30)
        self.switch_to_runner(check_switched=True)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.42)
        self.send_key_up("w")
        self.sleep(0.20)
        self.send_key_down("d")
        self.sleep(0.57)
        self.send_key("lshift", down_time=0.32)
        self.sleep(0.57)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_down("w")
        self.sleep(0.40)
        self.send_key("lshift", down_time=0.25)
        self.sleep(0.60)
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.2)
        self.sleep(0.10)
        self.switch_to_avoider(check_switched=True)  # 切到狗哥潜行避免碰到怪改变路径
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.15)
        self.send_key_down("lshift")
        self.sleep(0.24)
        self.send_key("d", down_time=0.30)
        self.sleep(1.28)
        self.send_key_up("lshift")
        self.sleep(0.64)
        self.send_key("d", down_time=0.12)
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(1.14)
        self.send_key_up("w")
        self.sleep(0.10)
        self.switch_to_fighter(check_switched=True, mode=1)  # 切到早雾控怪
        self.sleep(0.10)
        self.send_key("a", down_time=0.20)
        self.sleep(0.10)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        if self.find_interac():
            self.send_key("s", down_time=0.10)
            self.sleep(0.25)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key_down("e")
            self.sleep(0.10)
            self.send_key("e", down_time=2.40)
            self.send_key_down("w")
            self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        self.switch_to_avoider(check_switched=True)  # 切到狗哥潜行避免碰到怪改变路径
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key("d", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.42)
        self.sleep(0.76)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key("a", down_time=3.80)
        self.sleep(0.10)
        self.send_key("a", down_time=0.10)
        self.sleep(0.10)
        self.send_key("w", down_time=0.20)
        self.sleep(0.10)
        self.send_key("w", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=1.14)
        self.sleep(0.10)
        self.send_key("d", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=1.60)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.20)
        self.click(0.50, 0.50, key="middle", down_time=0.15)
        self.sleep(0.30)
        self.send_key_down("w")
        self.sleep(0.20)
        self.send_key("lshift", down_time=1.25)
        self.sleep(1.80)
        self.send_key("d", down_time=0.32)
        self.sleep(0.90)
        self.send_key_down("d")
        self.sleep(0.80)
        self.send_key_up("d")
        self.send_key_up("w")
        self.switch_to_fighter(check_switched=True, mode=1)  # 切到早雾控怪
        self.sleep(0.24)
        self.send_key("s", down_time=0.10)
        self.sleep(1.14)
        self.send_key("e", down_time=2.60)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.14)
        self.send_key_down("d")
        self.wait_and_interact(direction="d", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_up("w")
        if self.wait_ocr(x=0.60, y=0.52, to_x=0.70, to_y=0.57, match=re.compile("开门"), time_out=1.14):
            self.sleep(0.10)
            self.send_key("f", down_time=0.10)
            self.sleep(0.10)
        elif self.find_interac():
            self.sleep(0.20)
            self.clear_current_combat()
            self.sleep(0.10)
            self.send_key("w", down_time=0.32)
            self.sleep(0.10)
            self.send_key_down('d')
            self.sleep(0.05)
            self.wait_and_interact(direction="d", is_lock=False, time_out=3.65)
            if self.find_interac():
                is_open_door = self.lobby_open_door_check()
                if not is_open_door:
                    raise AbortException("timeout for wait_and_interact") # 考虑之后加复位或其他
                else:
                    self.sleep(0.10)
                    self.send_key("f", down_time=0.10)
                    self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("a", down_time=0.36)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def goto_lg1_skip_Hotori(self):
        self.log_round_info("浔、大厅前往LG1")
        self.sleep(0.30)
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.10)
        self.send_key_down('w')
        self.sleep(0.64)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.64)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.64)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.64)
        self.send_key_up('w')
        self.sleep(0.10)
        self.send_key_down('d')
        self.sleep(0.64)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.64)
        self.send_key_up('d')
        self.sleep(0.10)
        self.send_key_down('w')
        self.sleep(0.24)
        self.send_key("lshift", down_time=0.24)
        self.sleep(0.60)
        self.wait_and_interact(direction="w", is_lock=True, time_out=5.2)
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key_up("d")
        self.sleep(0.24)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.64)
        self.send_key('space', down_time=0.24)
        self.sleep(0.84)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.84)
        self.send_key_up("w")
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.send_key_down("w")
        self.wait_and_interact(direction="w", is_lock=True, time_out=6.4)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.32)
        self.send_key("a", down_time=0.32)
        self.sleep(0.32)
        self.send_key("a", down_time=0.42)
        self.sleep(0.76)
        self.send_key_up("w")
        self.sleep(0.10)
        self.send_key_down("a")
        self.sleep(0.20)
        self.send_key('lshift', down_time=0.20)
        self.sleep(0.20)
        self.send_key('lshift', down_time=0.20)
        self.sleep(0.20)
        self.send_key('lshift', down_time=0.20)
        self.sleep(1.20)
        self.send_key_up("a")
        self.sleep(0.10)
        self.send_key("a", down_time=0.10)
        self.sleep(0.10)
        self.send_key("w", down_time=0.20)
        self.sleep(0.10)
        self.send_key("w", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("d", down_time=0.15)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=1.16)
        self.sleep(0.10)
        self.send_key("d", down_time=0.10)
        self.sleep(0.10)
        self.send_key("d", down_time=1.82)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.10)
        self.send_key("s", down_time=0.10)
        self.sleep(0.20)
        self.click(0.50, 0.50, key="middle", down_time=0.15)
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down('w')
        self.sleep(1.14)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.42)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.42)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.42)
        self.send_key_down("d")
        self.sleep(0.42)
        self.send_key_up('d')
        self.sleep(0.42)
        self.send_key('lshift', down_time=0.24)
        self.sleep(0.76)
        self.send_key_up('w')
        self.sleep(0.10)
        self.click(down_time=0.64)
        self.sleep(0.10)
        self.send_key_down('w')
        self.wait_and_interact(direction="w", is_lock=True, time_out=7.64)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.24)
        self.send_key("a", down_time=0.42)
        self.wait_and_interact(direction="w", is_lock=False, time_out=3.65)
        self.sleep(0.30)

    def lobby_open_door_check(self, check_time = 3):
        open_door = False
        open_loop =0
        while not open_door and open_loop < check_time:
            if self.wait_ocr(x=0.60, y=0.52, to_x=0.70, to_y=0.57, match=re.compile("开门"), time_out=1.14):
                open_door = True
            else:
                self.sleep(0.10)
                self.send_key("f", down_time=0.10)
                self.sleep(0.20)
                open_loop += 1
        return open_door

    def lg1_wp1_safer(self):
        self.log_round_info("LG1 WP1 Safer")
        self.switch_to_runner(check_switched=True) # 确认切到薄荷跑图
        self.sleep(0.20)
        self.send_key('w', down_time=9.08)
        self.sleep(0.10)
        self.send_key('d', down_time=1.72)
        self.sleep(0.10)
        self.send_key('s', down_time=1.00)
        self.sleep(0.10)
        self.send_key('f', down_time=0.10) # 这里没必要上检测，门口不安全，停太久可能会被蚊子扫
        self.sleep(0.10)
        self.send_key('f', down_time=0.10)
        self.sleep(0.20)
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(1.25)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.03)
        self.send_key_up("d")
        self.sleep(0.22)
        self.send_key_down("a")
        self.sleep(3.90)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(0.40)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.01)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(5.60)
        self.send_key_up("d")
        self.sleep(0.06)
        self.send_key_down("w")
        self.sleep(2.02)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(3.21)
        self.send_key_up("d")
        self.sleep(0.12)

    def lg1_wp4_buster(self):
        self.log_round_info("LG1 WP4 bUSTER")
        self.send_key_down("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(3.31)
        self.send_key_up("s")
        self.sleep(0.12)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(1.50)
        self.send_key_up("w")
        self.sleep(0.11)
        self.start_interaction_watch()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.11)
        self.send_key_up("a")
        self.sleep(1.22)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(6.58)
        self.send_key_down("d")
        self.sleep(2.62)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.31)
        self.send_key_up("a")
        self.sleep(0.32)
        self.send_key_down("w")
        self.sleep(0.21)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.25)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(1.30)
        self.start_interaction_watch()
        self.send_key_down("d")
        self.sleep(2.10)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.65)
        self.send_key_down("w")
        self.sleep(0.22)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.48)
        self.send_key_down("space")
        self.sleep(0.14)
        self.send_key_up("space")
        self.sleep(0.14)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(1.41)
        self.stop_interaction_watch()
        self.send_key_down("w")
        self.sleep(0.81)
        self.send_key_up("w")
        self.sleep(0.11)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.47)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.60)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.38)
        self.send_key_up("w")
        self.sleep(0.34)
        self.send_key_down("d")
        self.sleep(0.61)
        self.send_key_up("d")
        self.sleep(0.11)
        self.loot_safes_while_walking(direction="s", time_out=2.37)
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.sleep(0.10)
        self.send_key_down("d")
        self.sleep(1.33)
        self.send_key_up("d")
        self.sleep(0.12)
        self.send_key_down("w")
        self.sleep(7.60)
        self.send_key_up("w")

    def lg1_wp5_buster(self):
        self.log_round_info("LG1 WP5 Buster 开始避战路线")
        self.switch_to_avoider(check_switched=True)
        self.sleep(0.50)
        self.perform_avoidance_action()
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(6.00)
        self.send_key_up("w")
        self.switch_to_runner(check_switched=True)
        self.sleep(0.32)
        self.wait_and_interact(is_lock=True)
        self.sleep(0.10)
        self.send_key_down("w")
        self.sleep(0.10)
        self.wait_and_interact(direction="w")

    def lg2_wp2_to_exit2_safer(self):
        self.log_round_info("LG2 WP2 Safer 尝试出口2")
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.21)
        self.send_key_down("space")
        self.sleep(0.10)
        self.send_key_up("space")
        self.sleep(0.80)
        self.send_key_up("d")
        self.sleep(0.20)
        self.send_key_up("f")  # end pick
        self.send_key_down('w')
        self.sleep(1.70)
        self.send_key_up('w')
        self.sleep(0.11)
        self.send_key_down('d')
        self.sleep(0.80)
        self.send_key("lshift", down_time=0.10)
        self.sleep(2.00)
        self.send_key_up('d')
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.31)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(0.81)
        self.send_key_up("d")
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.96)
        self.send_key_up("w")
        self.switch_to_runner()
        self.send_key_down("f")  # start pick
        self.sleep(0.11)
        self.send_key_down("a")
        self.sleep(0.71)
        self.send_key_up("a")
        self.sleep(0.31)
        self.send_key_down("d")
        self.sleep(1.61)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.20)
        self.send_key_down("a")
        self.sleep(0.72)
        self.send_key_up("a")
        self.sleep(1.26)
        self.send_key_down("w")
        self.sleep(2.60)
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(2.31)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(3.63)  # 4.03
        self.send_key_up("w")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(2.75)
        self.send_key_up("s")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("d")
        self.sleep(1.51)
        self.send_key_up("d")
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("s")
        self.sleep(0.60)
        self.send_key_up("s")
        self.sleep(0.11)
        self.send_key_up("f")  # end pick
        self.switch_to_runner()
        self.sleep(0.11)
        self.send_key_down("w")
        self.sleep(2.56)
        self.send_key_down("a")
        self.sleep(0.40)
        self.send_key_up("a")
        self.sleep(1.57)
        self.exit_state[2] = self.try_open_exit(direction="w")
        self.sleep(0.40)

    def check_current_floor_str(self, floor_str):
        ret = self.wait_ocr(0.04, 0.23, 0.17, 0.28, match=re.compile(floor_str), time_out=5)
        if ret:
            return True

    def switch_to_fighter(self, check_switched=False, mode="all_desc"):
        """切换到可用战斗角色。   
        `mode` 调度策略（配置重新从小到大排序后）：
        - "all_desc": [默认]按键位从大到小完整尝试（如 ["4", "1"]）
        - "all_asc" : 按键位从小到大完整尝试（如 ["1", "4"]）
        -     1     : 只切当前最小的那个键位
        -    -1     : 只切当前最大的那个键位
        -     n     : 只切重新排序后的【第 n 个】角色
        """
        config_keys = list(self.config.get(self.CONF_FIGHTER, []))
        if not config_keys:
            dead_keys = set(self._dead_fighter_keys)
            config_keys = [item for item in config_keys if item not in dead_keys]
            return self._begin_character_switch(self.ROLE_FIGHTER, config_keys, check_switched)
        sorted_keys = sorted(config_keys, key=int)
        if mode == "all_asc":
            keys = sorted_keys
        elif mode == "all_desc":
            keys = sorted_keys[::-1]
        elif isinstance(mode, int):   
            if mode == -1:
                keys = [sorted_keys[-1]]
            else:
                idx = mode - 1
                if 0 <= idx < len(sorted_keys):
                    keys = [sorted_keys[idx]]
                else:
                    self.log_error(f"切人位置越界！配置排序后只有 {len(sorted_keys)} 个人，你请求切第 {mode} 个，自动切最后一个。")
                    keys = [sorted_keys[-1]]
        else:
            keys = sorted_keys[::-1]
        dead_keys = set(self._dead_fighter_keys)
        keys = [item for item in keys if item not in dead_keys]
        return self._begin_character_switch(self.ROLE_FIGHTER, keys, check_switched)

@AgentServer.custom_action("PinkPawHeistScheme3Action")
class PinkPawHeistScheme3Action(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = _parse_custom_action_param(argv)
        path = PinkPawHeistCore3Path(context, params=params)
        try:
            path.log_round_info(
                f"Start copied OK-NTE route B, timing x{path.route_timing_scale:.2f}"
            )
            path.run_path()
            path._release_held_keys()
            path.ah.release_controls()
            path.exit_heist()
            return CustomAction.RunResult(success=True)
        except TaskerStoppedException as exc:
            print(f"[PinkPawHeist/Core3] stopped by tasker: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)
        except AbortException as exc:
            print(f"[PinkPawHeist/Core3] route aborted: {exc}")
            path._release_held_keys()
            path.abort_heist()
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            print(f"[PinkPawHeist/Core3] route failed: {exc}")
            path._release_held_keys()
            path.abort_heist()
            return CustomAction.RunResult(success=True)
