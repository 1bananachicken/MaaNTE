import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .rhythm.lanes import build_lane_layout, lane_full_roi_slice, lane_roi_quad
from .rhythm.detector import RhythmDetector
from .rhythm.presence import STATE_OTHER, STATE_PLAYING, STATE_RESULTS, STATE_SONG_SELECT, SceneGate
from .rhythm.song_selector import SongSelector

logger = logging.getLogger(__name__)

_LANES = ("d", "f", "j", "k")
_VK = {"d": 0x44, "f": 0x46, "j": 0x4A, "k": 0x4B}
_VK_ESCAPE = 0x1B

_DEFAULT_CONFIG = {
    "lanes": {
        "center_x_frac": [0.225, 0.406, 0.596, 0.771],
        "top_center_x_frac": [0.214, 0.406, 0.596, 0.783],
        "half_width_frac": 0.028,
        "judge_line_y_frac": 0.78,
        "judge_line_y_frac_by_lane": [0.78, 0.70, 0.78, 0.78],
        "judge_band_half_height_frac": 0.03,
        "judge_band_keep_from_top": 0.52,
        "roi_top_y_frac": 0.35,
    },
    "hsv_ranges": [
        {"name": "d", "h_min": 90, "h_max": 125, "s_min": 60, "s_max": 255, "v_min": 120, "v_max": 255},
        {"name": "f", "h_min": 18, "h_max": 38, "s_min": 95, "s_max": 255, "v_min": 165, "v_max": 255},
        {"name": "j", "h_min": 0, "h_max": 12, "s_min": 80, "s_max": 255, "v_min": 120, "v_max": 255, "h2_min": 165, "h2_max": 179},
        {"name": "k", "h_min": 125, "h_max": 165, "s_min": 50, "s_max": 255, "v_min": 100, "v_max": 255},
    ],
    "detection": {
        "enabled_lanes": [True, True, True, True],
        "component_mode_lanes": [True, False, True, True],
        "component_min_area_frac": 0.000043,
        "component_lookahead_y_frac": 0.083,
        "component_past_y_frac": 0.032,
        "component_same_note_y_frac": 0.032,
        "component_history_sec": 0.25,
        "min_pixels_per_lane": 200,
        "min_pixels_by_lane": [200, 200, 200, 200],
        "cooldown_sec": 0.12,
        "cooldown_sec_by_lane": [0.05, 0.05, 0.05, 0.05],
        "morph_kernel": 3,
        "log_cooldown_debug": False,
    },
    "presence": {
        "enabled": True,
        "drum_center_y_frac": 0.80,
        "patch_half_width_frac": 0.034,
        "min_laplace_variance": 95,
        "min_mean_gray": 18,
        "max_mean_gray": 255,
        "arm_after_good_frames": 1,
        "disarm_after_bad_frames": 5,
    },
    "scene": {
        "template_match_enabled": False,
        "state_confirm_frames": 2,
        "song_select_match_threshold": 0.75,
        "results_match_threshold": 0.75,
        "playing_match_threshold": 0.75,
        "match_vote_min": 2,
        "match_blur_ksize": 5,
        "match_downscale": 0.5,
        "match_skip_playing": True,
    },
    "song_select": {
        "enabled": False,
        "song_name": "",
        "scroll_area_x_frac": 0.35,
        "scroll_area_y_frac": 0.50,
        "scroll_delta": -3,
        "max_scroll_attempts": 30,
        "match_threshold": 0.75,
        "start_match_threshold": 0.75,
        "click_delay_sec": 0.5,
        "start_delay_sec": 0.8,
    },
    "auto_repeat": {
        "enabled": False,
        "count": 5,
        "dismiss_delay_sec": 0.8,
    },
    "keys": {
        "press_delay_sec": 0.0,
        "press_delay_sec_by_lane": [0.0, 0.08, 0.0, 0.0],
        "key_hold_sec": 0.01,
    },
    "run": {
        "target_fps": 60,
    },
}


def _load_rhythm_config() -> dict[str, Any]:
    cfg_paths = []
    current = Path(__file__).resolve().parents[3]
    cfg_paths.append(current / "assets" / "resource" / "base" / "pipeline" / "Rhythm" / "rhythm_config.json")
    cfg_paths.append(current / "resource" / "base" / "pipeline" / "Rhythm" / "rhythm_config.json")
    for p in cfg_paths:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                logger.info("已加载演奏配置文件: %s", p)
                return loaded
            except Exception:
                logger.warning("演奏配置文件读取失败: %s，使用内置默认值", p)
    logger.info("未找到外部演奏配置文件，使用内置默认值")
    return dict(_DEFAULT_CONFIG)


def _get_image(controller):
    job = controller.post_screencap()
    job.wait()
    return controller.cached_image


class _KeyDispatcher:
    """轻量异步按键分发器，使用独立线程保证时序精确。"""

    def __init__(self, cfg: dict[str, Any]):
        kc = cfg.get("keys") or {}
        self._press_delay_sec = float(kc.get("press_delay_sec", 0.0))
        raw_by = kc.get("press_delay_sec_by_lane")
        if isinstance(raw_by, list) and len(raw_by) >= 4:
            self._delay_by_lane = [float(raw_by[i]) for i in range(4)]
        else:
            self._delay_by_lane = [self._press_delay_sec] * 4
        self._key_hold_sec = float(kc.get("key_hold_sec", 0.01))

        self._lock = threading.Lock()
        self._queue: deque[tuple[float, list[int]]] = deque()
        self._event = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._fire_times: dict[int, float] = {}

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def dispatch(self, lane_indices: list[int], target_time: float):
        with self._lock:
            self._queue.append((target_time, list(lane_indices)))
        self._event.set()

    def drain_fire_times(self) -> dict[int, float]:
        with self._lock:
            result = dict(self._fire_times)
            self._fire_times.clear()
        return result

    def stop(self):
        self._stopping.set()
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _worker(self):
        while not self._stopping.is_set():
            self._event.wait(timeout=0.005)
            self._event.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    target_time, lanes = self._queue[0]
                now = time.perf_counter()
                if now < target_time:
                    time.sleep(max(0, target_time - now - 0.001))
                with self._lock:
                    if self._queue and self._queue[0][0] == target_time:
                        self._queue.popleft()
                self._execute(lanes, target_time)

    def _execute(self, lanes: list[int], target_time: float):
        import win32api
        import win32con
        import win32gui
        hwnd = self._game_hwnd if self._game_hwnd else 0
        act_hwnd = hwnd or 0
        for li in lanes:
            delay = self._delay_by_lane[li]
            if delay > 0:
                time.sleep(delay)
            key = _VK[_LANES[li]]
            if act_hwnd:
                win32api.PostMessage(act_hwnd, win32con.WM_KEYDOWN, key, 0)
            else:
                win32api.keybd_event(key, 0, 0, 0)
        time.sleep(self._key_hold_sec)
        for li in lanes:
            key = _VK[_LANES[li]]
            if act_hwnd:
                win32api.PostMessage(act_hwnd, win32con.WM_KEYUP, key, 0)
            else:
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)
        with self._lock:
            for li in lanes:
                self._fire_times[li] = target_time

    def set_game_hwnd(self, hwnd: int):
        self._game_hwnd = hwnd

    _game_hwnd: int = 0


def _find_game_hwnd():
    import win32gui
    titles = ["异环", "NTE"]
    for title in titles:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            return hwnd
    for title in titles:
        def enum_callback(hwnd, results):
            text = win32gui.GetWindowText(hwnd)
            if title in text:
                results.append(hwnd)
            return True
        results: list[int] = []
        win32gui.EnumWindows(enum_callback, results)
        if results:
            return results[0]
    return 0


def _do_scroll(hwnd: int, x: int, y: int, delta: int):
    import win32api
    import win32con
    lparam = (y << 16) | (x & 0xFFFF)
    if hwnd:
        win32api.PostMessage(hwnd, win32con.WM_MOUSEWHEEL, (delta * 120) << 16, lparam)


@AgentServer.custom_action("auto_rhythm")
class AutoRhythm(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller
        cfg = _load_rhythm_config()

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        if "song_name" in params:
            cfg.setdefault("song_select", {})["enabled"] = True
            cfg.setdefault("song_select", {})["song_name"] = str(params["song_name"])
        if "auto_repeat_count" in params:
            cfg.setdefault("auto_repeat", {})["enabled"] = True
            cfg.setdefault("auto_repeat", {})["count"] = int(params["auto_repeat_count"])
        if "target_fps" in params:
            cfg.setdefault("run", {})["target_fps"] = int(params["target_fps"])

        game_hwnd = _find_game_hwnd()
        logger.info("游戏窗口 HWND: %d", game_hwnd)

        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)

        detector = RhythmDetector(cfg)
        scene_gate = SceneGate(cfg)
        song_selector = SongSelector(cfg)

        dispatcher = _KeyDispatcher(cfg)
        dispatcher.set_game_hwnd(game_hwnd)
        dispatcher.start()

        auto_repeat_cfg = cfg.get("auto_repeat") or {}
        auto_repeat_enabled = bool(auto_repeat_cfg.get("enabled", False))
        auto_repeat_count = int(auto_repeat_cfg.get("count", 1))
        auto_repeat_dismiss_delay = float(auto_repeat_cfg.get("dismiss_delay_sec", 0.8))
        repeat_index = 0
        results_seen = False

        try:
            while True:
                if context.tasker.stopping:
                    logger.info("任务被中断，停止演奏")
                    break

                t0 = time.perf_counter()

                frame = _get_image(controller)
                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                fh, fw = frame.shape[:2]
                if fh <= 0 or fw <= 0:
                    time.sleep(0.1)
                    continue

                layout = build_lane_layout(cfg, fw, fh)
                state, gate_info = scene_gate.step(frame, layout)

                if state == STATE_SONG_SELECT and song_selector.enabled:
                    scroll_fn = None if game_hwnd == 0 else lambda sx, sy, sd: _do_scroll(game_hwnd, sx, sy, sd)
                    sel_info = song_selector.step(frame, layout, controller, scroll_func=scroll_fn)
                    sel_state = sel_info.get("state", "")
                    if sel_state in ("waiting", "done"):
                        results_seen = False
                    if sel_state == "failed":
                        logger.warning("自动选歌失败")
                elif state == STATE_PLAYING:
                    song_selector.reset()
                    results_seen = False
                else:
                    song_selector.reset()

                if auto_repeat_enabled and state == STATE_RESULTS:
                    if not results_seen:
                        results_seen = True
                        repeat_index += 1
                        logger.info("检测到结算界面: 第 %d/%d 次", repeat_index, auto_repeat_count)
                    if repeat_index >= auto_repeat_count:
                        logger.info("已达到连打次数上限 (%d)，停止", auto_repeat_count)
                        time.sleep(auto_repeat_dismiss_delay)
                        dispatcher.stop()
                        break
                    time.sleep(auto_repeat_dismiss_delay)
                    import win32api
                    import win32con
                    if game_hwnd:
                        win32api.PostMessage(game_hwnd, win32con.WM_KEYDOWN, _VK_ESCAPE, 0)
                        time.sleep(0.05)
                        win32api.PostMessage(game_hwnd, win32con.WM_KEYUP, _VK_ESCAPE, 0)
                    else:
                        controller.post_click_key(_VK_ESCAPE).wait()
                    time.sleep(1.0)
                    results_seen = False
                    song_selector.reset()
                    continue

                triggers, masks, pixels = detector.analyze(frame, layout)

                if state != STATE_PLAYING:
                    triggers = [False, False, False, False]

                armed = gate_info.get("armed", gate_info.get("enabled", True))
                if not armed:
                    triggers = [False, False, False, False]

                triggered_lanes = [i for i, t in enumerate(triggers) if t]
                if triggered_lanes:
                    fire_times = dispatcher.drain_fire_times()
                    detector.update_fire_times(fire_times)
                    now = time.perf_counter()
                    press_delay = float(cfg.get("keys", {}).get("press_delay_sec", 0.0))
                    dispatcher.dispatch(triggered_lanes, now + press_delay)

                fire_times = dispatcher.drain_fire_times()
                if fire_times:
                    detector.update_fire_times(fire_times)

                elapsed = time.perf_counter() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        finally:
            dispatcher.stop()
            logger.info("演奏自动化已停止")

        return CustomAction.RunResult(success=True)
