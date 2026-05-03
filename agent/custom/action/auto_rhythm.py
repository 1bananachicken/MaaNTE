import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JOCR

from .rhythm.utils.lanes import build_lane_layout
from .rhythm.utils.detector import DrumDetector
from .rhythm.utils.presence import (
    STATE_OTHER,
    STATE_PLAYING,
    STATE_RESULTS,
    STATE_SONG_SELECT,
    SceneGate,
)
from .rhythm.utils.song_selector import SongSelector


logger = logging.getLogger(__name__)

_LANES = ("d", "f", "j", "k")
_VK = {"d": 0x44, "f": 0x46, "j": 0x4A, "k": 0x4B}
_VK_ESCAPE = 0x1B

_RHYTHM_ENV_KEYS = (
    "RHYTHM_AUTO_SELECT",
    "RHYTHM_SONG_NAME",
    "RHYTHM_AUTO_REPEAT_COUNT",
    "RHYTHM_TARGET_FPS",
    "RHYTHM_AUTO_REPEAT_MAX",
)

_DEFAULT_CONFIG = {
    "lanes": {
        "center_x_frac": [0.225, 0.406, 0.596, 0.771],
        "top_center_x_frac": [0.214, 0.406, 0.596, 0.783],
        "half_width_frac": 0.028,
        "judge_line_y_frac": 0.78,
        "judge_line_y_frac_by_lane": [0.78, 0.78, 0.78, 0.78],
        "judge_band_half_height_frac": 0.03,
    },
    "template_detection": {
        "thresholds": [0.81, 0.80, 0.80, 0.81],
        "cooldown_sec": 0.05,
        "cooldown_sec_by_lane": [0.05, 0.05, 0.05, 0.05],
        "region_extend_up_frac": 0.14,
        "region_extend_down_frac": 0.15,
        "region_width_multiplier": 4.0,
        "enabled_lanes": [True, True, True, True],
    },
    "scene": {
        "state_confirm_frames": 1,
        "song_select_match_threshold": 0.75,
        "results_match_threshold": 0.75,
        "playing_match_threshold": 0.75,
        "match_vote_min": 1,
        "playing_check_interval": 5,
        "scene_lock_timeout_sec": 1.5,
        "song_select_to_playing_lock_sec": 8.0,
    },
    "song_select": {
        "enabled": False,
        "song_name": "",
        "scroll_area_x_frac": 0.25,
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
        "press_delay_sec_by_lane": [0.0, 0.0, 0.0, 0.0],
        "key_hold_sec": 0.01,
    },
    "run": {
        "target_fps": 60,
    },
}


def _load_rhythm_config() -> dict[str, Any]:
    here = Path(__file__).resolve()
    cfg_paths = []
    for i in range(len(here.parents)):
        root = here.parents[i]
        cfg_paths.append(root / "resource" / "base" / "rhythm_config.json")
        cfg_paths.append(root / "assets" / "resource" / "base" / "rhythm_config.json")
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


def _press_keys(controller, lane_indices: list[int], key_hold_sec: float = 0.01):
    for li in lane_indices:
        controller.post_key_down(_VK[_LANES[li]])
    time.sleep(key_hold_sec)
    for li in lane_indices:
        controller.post_key_up(_VK[_LANES[li]])


_COST_VITALITY_ROI = (540, 594, 187, 35)
_COST_VITALITY_PATTERN = re.compile(r"-\s*(\d+)")


def _detect_cost_vitality(context: Context, frame) -> int:
    if frame is None or frame.size == 0:
        return 0

    detail = context.run_recognition_direct(
        JRecognitionType.OCR, JOCR(roi=_COST_VITALITY_ROI), frame
    )
    if detail is None or not detail.hit or detail.best_result is None:
        logger.debug("消耗活力 OCR 未命中")
        return 0

    text = detail.best_result.text if hasattr(detail.best_result, "text") else str(detail.best_result)
    m = _COST_VITALITY_PATTERN.search(text)
    if m:
        cost = int(m.group(1))
        logger.debug("消耗活力 OCR: %s -> %d", text, cost)
        return cost

    logger.debug("消耗活力 OCR 未匹配: %s", text)
    return 0


@AgentServer.custom_action("rhythm_set_param")
class RhythmSetParam(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                params = {}
            for k, v in params.items():
                os.environ[f"RHYTHM_{k.upper()}"] = str(v)
            logger.debug(
                "rhythm_set_param: %s",
                {k: os.environ.get(f"RHYTHM_{k.upper()}") for k in params},
            )
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("auto_rhythm")
class AutoRhythm(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        cfg = _load_rhythm_config()

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                pass

        merged = dict(params)
        for env_key in _RHYTHM_ENV_KEYS:
            val = os.environ.get(env_key)
            if val is None:
                continue
            if env_key.endswith("_AUTO_SELECT"):
                merged.setdefault("auto_select", val.lower() in ("true", "1", "yes"))
            elif env_key.endswith("_SONG_NAME"):
                merged.setdefault("song_name", val)
            elif env_key.endswith("_AUTO_REPEAT_COUNT"):
                merged.setdefault("auto_repeat_count", val)
            elif env_key.endswith("_TARGET_FPS"):
                merged.setdefault("target_fps", val)
            elif env_key.endswith("_AUTO_REPEAT_MAX"):
                merged.setdefault("auto_repeat_max", val)

        if "auto_select" in merged:
            cfg.setdefault("song_select", {})["auto_select"] = merged["auto_select"]
            if cfg["song_select"]["auto_select"]:
                cfg["song_select"]["enabled"] = True
        if "song_name" in merged:
            cfg.setdefault("song_select", {})["song_name"] = str(merged["song_name"])
            if cfg["song_select"]["song_name"]:
                cfg["song_select"]["enabled"] = True
        if "auto_repeat_count" in merged:
            cfg.setdefault("auto_repeat", {})["count"] = int(
                merged["auto_repeat_count"]
            )
            if cfg["auto_repeat"]["count"] > 0:
                cfg["auto_repeat"]["enabled"] = True
        if "auto_repeat_max" in merged:
            cfg.setdefault("auto_repeat", {})["max"] = (
                str(merged["auto_repeat_max"]).lower() in ("true", "1", "yes")
            )
            if cfg["auto_repeat"]["max"]:
                cfg["auto_repeat"]["enabled"] = True
        if "target_fps" in merged:
            cfg.setdefault("run", {})["target_fps"] = int(merged["target_fps"])

        cfg.setdefault("scene", {}).setdefault("playing_check_interval", 5)
        cfg.setdefault("scene", {}).setdefault("state_confirm_frames", 1)
        cfg.setdefault("scene", {}).setdefault("scene_lock_timeout_sec", 1.5)

        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)

        key_hold_sec = float(cfg.get("keys", {}).get("key_hold_sec", 0.01))

        scene_gate = SceneGate(cfg)
        song_selector = SongSelector(cfg)

        detector = DrumDetector(cfg)
        drum_available = detector.available
        if not drum_available:
            logger.warning("鼓面模板图片缺失，演奏检测将不可用")

        auto_repeat_cfg = cfg.get("auto_repeat") or {}
        auto_repeat_enabled = bool(auto_repeat_cfg.get("enabled", False))
        auto_repeat_count = int(auto_repeat_cfg.get("count", 1))
        auto_repeat_max = bool(auto_repeat_cfg.get("max", False))
        auto_repeat_dismiss_delay = float(auto_repeat_cfg.get("dismiss_delay_sec", 0.8))

        scene_lock_timeout = float(
            cfg.get("scene", {}).get("scene_lock_timeout_sec", 1.5)
        )
        song_select_to_playing_lock = float(
            cfg.get("scene", {}).get("song_select_to_playing_lock_sec", 8.0)
        )

        logger.info(
            "演奏任务开始 | 目标FPS=%d | 鼓面检测=%s | 自动选歌=%s(%s) | 自动连打=%s(次数=%d|Max=%s)",
            target_fps,
            drum_available,
            song_selector.enabled,
            song_selector.song_name,
            auto_repeat_enabled,
            auto_repeat_count,
            auto_repeat_max,
        )
        repeat_index = 0
        results_seen = False
        esc_sent_for_results = False
        cached_layout: Any = None
        cached_layout_size: tuple[int, int] = (0, 0)

        frame_count = 0
        playing_frame_count = 0
        prev_logged_state: str | None = None
        state = STATE_OTHER
        scene_lock_until: float = 0.0
        scene_lock_duration: float = scene_lock_timeout

        try:
            while True:
                if context.tasker.stopping:
                    logger.info("tasker 发出停止信号，退出演奏循环")
                    break

                t0 = time.perf_counter()

                controller.post_screencap().wait()
                frame = controller.cached_image
                if frame is None or frame.size == 0:
                    time.sleep(0.1)
                    continue

                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                fh, fw = frame.shape[:2]
                if fh <= 0 or fw <= 0:
                    time.sleep(0.1)
                    continue

                frame_count += 1

                state_before_step = state

                if state == STATE_PLAYING:
                    now = time.perf_counter()
                    if now < scene_lock_until:
                        pass
                    else:
                        new_state, _ = scene_gate.step(frame)
                        if new_state != state:
                            state = new_state
                else:
                    state, gate_info = scene_gate.step(frame)

                if state == STATE_PLAYING and state_before_step != STATE_PLAYING:
                    if state_before_step == STATE_SONG_SELECT:
                        scene_lock_duration = song_select_to_playing_lock
                    else:
                        scene_lock_duration = scene_lock_timeout
                    scene_lock_until = time.perf_counter() + scene_lock_duration

                if state != prev_logged_state:
                    logger.info(
                        "场景切换: %s -> %s",
                        prev_logged_state or "启动",
                        state,
                    )
                    prev_logged_state = state

                if state == STATE_SONG_SELECT:
                    scroll_fn = lambda sx, sy, sd: (
                        controller.post_swipe(sx, sy, sx, sy + sd * 100, duration=300).wait(),
                        time.sleep(0.1),
                    )
                    sel_info = song_selector.step(
                        frame, controller, scroll_func=scroll_fn
                    )
                    sel_state = sel_info.get("state", "")
                    if sel_state == "done":
                        logger.info("选歌完成，等待进入演奏界面")
                        results_seen = False
                    elif sel_state == "failed":
                        logger.warning("自动选歌失败")
                    time.sleep(frame_interval)
                    continue

                elif state == STATE_PLAYING:
                    song_selector.reset()
                    results_seen = False
                    esc_sent_for_results = False
                    playing_frame_count += 1

                    if drum_available:
                        if (fw, fh) != cached_layout_size:
                            cached_layout = build_lane_layout(cfg, fw, fh)
                            cached_layout_size = (fw, fh)
                        triggers, scores = detector.analyze(frame, cached_layout)
                        triggered_lanes = [i for i, t in enumerate(triggers) if t]
                        if triggered_lanes:
                            scene_lock_until = time.perf_counter() + scene_lock_duration
                            lane_names = [_LANES[i] for i in triggered_lanes]
                            # logger.debug(
                            #     "触发按键: %s | 帧#%d | scores=%s",
                            #     lane_names,
                            #     frame_count,
                            #     [f"{scores[i]:.3f}" for i in triggered_lanes],
                            # )
                            _press_keys(controller, triggered_lanes, key_hold_sec)

                elif state == STATE_RESULTS:
                    song_selector.reset()
                    playing_frame_count = 0
                    if not results_seen:
                        results_seen = True
                        repeat_index += 1
                        esc_sent_for_results = False
                        logger.info(
                            "检测到结算界面: 第 %d/%d 次",
                            repeat_index,
                            auto_repeat_count if not auto_repeat_max else -1,
                        )

                    if not esc_sent_for_results:
                        time.sleep(auto_repeat_dismiss_delay)
                        logger.info(
                            "发送 ESC 退出结算界面 (第 %d/%d 次)",
                            repeat_index,
                            auto_repeat_count if auto_repeat_enabled else 1,
                        )
                        controller.post_click_key(_VK_ESCAPE).wait()
                        esc_sent_for_results = True

                    if not auto_repeat_enabled:
                        logger.info(
                            "自动连打未启用，已完成第 %d 次演奏，停止",
                            repeat_index,
                        )
                        time.sleep(auto_repeat_dismiss_delay)
                        break

                    if auto_repeat_max:
                        cost = _detect_cost_vitality(context, frame)
                        if cost == 0:
                            logger.info(
                                "检测到消耗活力为 0，活力已耗尽，停止连打"
                            )
                            time.sleep(auto_repeat_dismiss_delay)
                            break
                        logger.info(
                            "检测到消耗活力为 %d，继续连打", cost
                        )
                    elif repeat_index >= auto_repeat_count:
                        logger.info("已达到连打次数上限 (%d)，停止", auto_repeat_count)
                        time.sleep(auto_repeat_dismiss_delay)
                        break

                    time.sleep(1.0)
                    continue

                else:
                    song_selector.reset()
                    playing_frame_count = 0

                # if frame_count % 120 == 0:
                #     logger.debug(
                #         "帧#%d | state=%s | playing_frames=%d",
                #         frame_count,
                #         state,
                #         playing_frame_count,
                #     )

                elapsed = time.perf_counter() - t0
                if state != STATE_PLAYING and elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        finally:
            for env_key in _RHYTHM_ENV_KEYS:
                os.environ.pop(env_key, None)
            logger.info("演奏自动化已停止")

        return CustomAction.RunResult(success=True)
