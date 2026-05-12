import json
import logging
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from ..utils.config import load_rhythm_config
from ..utils.lanes import build_lane_layout, LaneLayout
from ..utils.detector import DrumDetector
from ..utils.assets import list_scene_templates, read_image
from ..utils.presence import SceneGate, STATE_PLAYING


logger = logging.getLogger(__name__)

_LANES = ("d", "f", "j", "k")
_VK = {"d": 0x44, "f": 0x46, "j": 0x4A, "k": 0x4B}

_TIMEOUT_SEC = 600
_SCENE_LOCK_SEC = 8.0
_PLAYING_CHECK_THRESHOLD = 0.7
_LANE_COUNT = 4


class _KeyScheduler:
    def __init__(
        self,
        controller: Any,
        key_hold_sec: float,
        chord_reinforce_count: int,
        chord_reinforce_interval_sec: float,
        duplicate_window_sec: float,
        chord_window_sec: float,
        min_tap_interval_by_lane: list[float],
    ) -> None:
        self._controller = controller
        self._key_hold_sec = max(0.001, key_hold_sec)
        self._chord_reinforce_count = max(0, chord_reinforce_count)
        self._chord_reinforce_interval_sec = max(0.0, chord_reinforce_interval_sec)
        self._duplicate_window_sec = max(0.001, duplicate_window_sec)
        self._chord_window_sec = max(0.001, chord_window_sec)
        self._min_tap_interval_by_lane = min_tap_interval_by_lane
        self._active_until = [0.0] * _LANE_COUNT
        self._is_down = [False] * _LANE_COUNT
        self._last_tap_time = [-999.0] * _LANE_COUNT
        self._recent_target_times: list[list[float]] = [[] for _ in range(_LANE_COUNT)]
        self._pending: list[tuple[float, int, float, float, float]] = []

    def press(self, lane_indices: list[int], now: float | None = None) -> None:
        lanes = sorted(set(lane_indices))
        if not lanes:
            return

        now = time.perf_counter() if now is None else now
        self.release_expired(now)

        for li in lanes:
            if self._is_down[li]:
                self._controller.post_key_up(_VK[_LANES[li]]).wait()
                self._is_down[li] = False

        jobs = [self._controller.post_key_down(_VK[_LANES[li]]) for li in lanes]
        for job in jobs:
            job.wait()

        if len(lanes) > 1:
            for _ in range(self._chord_reinforce_count):
                if self._chord_reinforce_interval_sec > 0:
                    time.sleep(self._chord_reinforce_interval_sec)
                jobs = [self._controller.post_key_down(_VK[_LANES[li]]) for li in lanes]
                for job in jobs:
                    job.wait()

        hold_until = time.perf_counter() + self._key_hold_sec
        for li in lanes:
            self._is_down[li] = True
            self._active_until[li] = hold_until
            self._last_tap_time[li] = now

    def schedule(
        self,
        lane_idx: int,
        due_time: float,
        target_time: float,
        center_y: float,
        score: float,
        now: float,
    ) -> bool:
        self._prune_recent(now)
        if any(
            abs(target_time - recent) <= self._duplicate_window_sec
            for recent in self._recent_target_times[lane_idx]
        ):
            return False
        if any(
            lane_idx == pending_lane
            and abs(target_time - pending_target) <= self._duplicate_window_sec
            for _, pending_lane, pending_target, _, _ in self._pending
        ):
            return False

        self._recent_target_times[lane_idx].append(target_time)
        self._pending.append((due_time, lane_idx, target_time, center_y, score))
        return True

    def fire_due(self, now: float) -> list[tuple[int, float, float, float]]:
        if not self._pending:
            return []

        self._pending.sort(key=lambda item: item[0])
        anchor_item = next((item for item in self._pending if item[0] <= now), None)
        if anchor_item is None:
            return []

        anchor_due, _, anchor_target, _, _ = anchor_item
        due: list[tuple[float, int, float, float, float]] = []
        remaining: list[tuple[float, int, float, float, float]] = []
        for item in self._pending:
            due_time, _, target_time, _, _ = item
            same_chord = (
                abs(target_time - anchor_target) <= self._chord_window_sec
                or due_time <= anchor_due + self._chord_window_sec
            )
            if same_chord:
                due.append(item)
            else:
                remaining.append(item)
        self._pending = remaining

        lanes: list[int] = []
        fired: list[tuple[int, float, float, float]] = []
        for due_time, lane_idx, target_time, center_y, score in due:
            next_allowed_time = self._last_tap_time[lane_idx] + self._min_tap_interval_by_lane[lane_idx]
            if now < next_allowed_time:
                logger.debug(
                    "候选丢弃: lane=%s target=%.3f next_allowed=%.3f y=%.1f score=%.3f reason=min_interval",
                    _LANES[lane_idx],
                    target_time,
                    next_allowed_time,
                    center_y,
                    score,
                )
                continue
            lanes.append(lane_idx)
            fired.append((lane_idx, target_time, center_y, score))

        if lanes:
            self.press(lanes, now)
        return fired

    def release_expired(self, now: float) -> None:
        lanes = [
            li
            for li in range(_LANE_COUNT)
            if self._is_down[li] and now >= self._active_until[li]
        ]
        jobs = [self._controller.post_key_up(_VK[_LANES[li]]) for li in reversed(lanes)]
        for job in jobs:
            job.wait()
        for li in lanes:
            self._is_down[li] = False

    def _prune_recent(self, now: float) -> None:
        keep_after = now - 0.5
        for i, times in enumerate(self._recent_target_times):
            self._recent_target_times[i] = [target_time for target_time in times if target_time >= keep_after]

    def close(self) -> None:
        if self._pending:
            logger.debug("候选丢弃: count=%d reason=scheduler_close", len(self._pending))
            self._pending.clear()
        lanes = [li for li in range(_LANE_COUNT) if self._is_down[li]]
        jobs = [self._controller.post_key_up(_VK[_LANES[li]]) for li in reversed(lanes)]
        for job in jobs:
            job.wait()
        for li in lanes:
            self._is_down[li] = False


def _load_playing_check_template() -> NDArray[np.uint8] | None:
    templates = list_scene_templates("playing")
    for name, path in templates:
        if name == "pause":
            img = read_image(path)
            if img is not None:
                logger.info(
                    "已加载演奏状态检测模板: pause.png (%dx%d)",
                    img.shape[1],
                    img.shape[0],
                )
                return img
    logger.warning("未找到演奏状态检测模板 (pause.png)")
    return None


def _is_still_playing(
    frame: NDArray[np.uint8], template: NDArray[np.uint8] | None
) -> bool:
    if template is None or frame is None or frame.size == 0:
        return True
    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    if th > fh or tw > fw:
        return True
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= _PLAYING_CHECK_THRESHOLD


def _normalize_same_frame_chords(
    events: list[tuple[int, float, float, float]],
    window_sec: float,
) -> list[tuple[int, float, float, float]]:
    if window_sec <= 0 or len(events) <= 1:
        return events

    normalized: list[tuple[int, float, float, float]] = []
    groups: list[list[tuple[int, float, float, float]]] = []
    for event in sorted(events, key=lambda item: item[1]):
        lane_idx, target_time, _, _ = event
        for group in groups:
            group_lanes = {item[0] for item in group}
            group_anchor = sum(item[1] for item in group) / len(group)
            if lane_idx not in group_lanes and abs(target_time - group_anchor) <= window_sec:
                group.append(event)
                merged = True
                break
        else:
            merged = False
        if not merged:
            groups.append([event])

    for group in groups:
        if len(group) == 1:
            normalized.extend(group)
            continue
        normalized_target_time = sum(item[1] for item in group) / len(group)
        for lane_idx, _, center_y, score in group:
            normalized.append((lane_idx, normalized_target_time, center_y, score))

    return normalized


@AgentServer.custom_action("auto_rhythm_play")
class AutoRhythmPlay(CustomAction):
    _playing_check_template: NDArray[np.uint8] | None = None
    _template_loaded: bool = False

    @classmethod
    def _ensure_template_loaded(cls) -> None:
        if cls._template_loaded:
            return
        cls._template_loaded = True
        cls._playing_check_template = _load_playing_check_template()

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self._ensure_template_loaded()
        controller = context.tasker.controller
        cfg = load_rhythm_config()

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except Exception:
                params = {}
            if "target_fps" in params:
                cfg.setdefault("run", {})["target_fps"] = int(params["target_fps"])

        target_fps = int(cfg.get("run", {}).get("target_fps", 60))
        frame_interval = 1.0 / max(1, target_fps)
        key_cfg = cfg.get("keys") or {}
        key_hold_sec = float(key_cfg.get("key_hold_sec", 0.01))
        chord_reinforce_count = int(key_cfg.get("chord_reinforce_count", 1))
        chord_reinforce_interval_sec = float(key_cfg.get("chord_reinforce_interval_sec", 0.006))
        thresholds = list(
            (cfg.get("template_detection") or {}).get(
                "thresholds", [0.81, 0.80, 0.80, 0.81]
            )
        )
        if len(thresholds) < 4:
            thresholds.extend([0.80] * (4 - len(thresholds)))
        simultaneous_margin = max(
            0.0,
            float((cfg.get("template_detection") or {}).get("simultaneous_score_margin", 0.04)),
        )
        trigger_cfg = cfg.get("position_trigger") or {}
        trigger_offset_frac = float(trigger_cfg.get("trigger_line_offset_frac", -0.012))
        trigger_band_frac = max(0.0, float(trigger_cfg.get("trigger_band_half_height_frac", 0.018)))
        min_tap_interval_sec = float(trigger_cfg.get("min_tap_interval_sec", 0.035))
        min_tap_by_lane = trigger_cfg.get("min_tap_interval_sec_by_lane")
        if isinstance(min_tap_by_lane, list) and len(min_tap_by_lane) == _LANE_COUNT:
            min_tap_interval_by_lane = [float(x) for x in min_tap_by_lane]
        else:
            min_tap_interval_by_lane = [min_tap_interval_sec] * _LANE_COUNT
        note_speed_px_per_sec = max(
            1.0,
            float(trigger_cfg.get("note_speed_px_per_sec", 900.0)),
        )
        input_latency_sec = max(
            0.0,
            float(trigger_cfg.get("input_latency_sec", 0.035)),
        )
        schedule_window_sec = max(
            0.0,
            float(trigger_cfg.get("schedule_window_sec", 0.22)),
        )
        stale_note_sec = max(
            0.0,
            float(trigger_cfg.get("stale_note_sec", 0.045)),
        )
        duplicate_window_sec = max(
            0.001,
            float(trigger_cfg.get("duplicate_window_sec", 0.025)),
        )
        chord_window_sec = max(
            0.001,
            float(trigger_cfg.get("chord_window_sec", 0.022)),
        )
        same_frame_chord_window_sec = max(
            0.0,
            float(trigger_cfg.get("same_frame_chord_window_sec", 0.045)),
        )
        debug_score_interval = max(
            0,
            int(cfg.get("run", {}).get("debug_score_interval_frames", target_fps)),
        )

        scene_lock_sec = float(
            cfg.get("scene", {}).get("song_select_to_playing_lock_sec", _SCENE_LOCK_SEC)
        )

        detector = DrumDetector(cfg)
        drum_available = detector.available
        if not drum_available:
            logger.warning("鼓面模板缺失，演奏检测不可用")

        scene_gate = SceneGate(cfg)

        logger.info(
            "演奏开始 | FPS=%d | 鼓面检测=%s | 场景冷却=%.1fs(音符命中重置)",
            target_fps, drum_available, scene_lock_sec,
        )

        start_time = time.perf_counter()
        frame_count = 0
        cached_layout: LaneLayout | None = None
        cached_layout_size: tuple[int, int] = (0, 0)
        playing_tpl = self._playing_check_template

        scene_lock_until = time.perf_counter() + scene_lock_sec
        key_scheduler = _KeyScheduler(
            controller,
            key_hold_sec,
            chord_reinforce_count,
            chord_reinforce_interval_sec,
            duplicate_window_sec,
            chord_window_sec,
            min_tap_interval_by_lane,
        )

        def fire_and_log(now: float) -> None:
            nonlocal scene_lock_until
            fired_notes = key_scheduler.fire_due(now)
            if not fired_notes:
                return
            scene_lock_until = now + scene_lock_sec
            logger.debug(
                "节拍触发: lanes=%s target=%s y=%s score=%s",
                [_LANES[i] for i, _, _, _ in fired_notes],
                ["%.3f" % target_time for _, target_time, _, _ in fired_notes],
                ["%.1f" % center_y for _, _, center_y, _ in fired_notes],
                ["%.3f" % score for _, _, _, score in fired_notes],
            )

        try:
            while True:
                now = time.perf_counter()
                # Tick before capture and after frame pacing so scheduled taps fire close to their target time.
                key_scheduler.release_expired(now)
                fire_and_log(now)
                if context.tasker.stopping:
                    logger.info("tasker 停止信号，退出演奏")
                    return CustomAction.RunResult(success=False)

                elapsed_total = time.perf_counter() - start_time
                if elapsed_total > _TIMEOUT_SEC:
                    logger.warning("演奏超时 (%d秒)", _TIMEOUT_SEC)
                    return CustomAction.RunResult(success=False)

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
                now = time.perf_counter()

                if drum_available:
                    if (fw, fh) != cached_layout_size:
                        cached_layout = build_lane_layout(cfg, fw, fh)
                        cached_layout_size = (fw, fh)
                    scores, candidates_by_lane = detector.analyze(frame, cached_layout)
                    candidate_events: list[tuple[int, float, float, float]] = []
                    scheduled: list[tuple[int, float, float, float]] = []
                    for i in range(_LANE_COUNT):
                        judge_center_y = (
                            cached_layout.judge_y0_by_lane[i] + cached_layout.judge_y1_by_lane[i]
                        ) / 2.0
                        trigger_y = judge_center_y + trigger_offset_frac * fh
                        trigger_half_band = trigger_band_frac * fh
                        latest_acceptable_y = trigger_y + trigger_half_band
                        score_threshold = float(thresholds[i]) - simultaneous_margin

                        for candidate in candidates_by_lane[i]:
                            if candidate.score < score_threshold:
                                continue
                            if candidate.center_y > latest_acceptable_y:
                                continue

                            eta_sec = (trigger_y - candidate.center_y) / note_speed_px_per_sec
                            target_time = now + eta_sec
                            if target_time < now - stale_note_sec:
                                continue

                            due_time = max(now, target_time - input_latency_sec)
                            if due_time > now + schedule_window_sec:
                                continue

                            candidate_events.append((i, target_time, candidate.center_y, candidate.score))

                    for i, target_time, center_y, score in _normalize_same_frame_chords(
                        candidate_events,
                        same_frame_chord_window_sec,
                    ):
                        due_time = max(now, target_time - input_latency_sec)
                        if due_time > now + schedule_window_sec:
                            continue

                        if key_scheduler.schedule(
                            lane_idx=i,
                            due_time=due_time,
                            target_time=target_time,
                            center_y=center_y,
                            score=score,
                            now=now,
                        ):
                            scheduled.append((i, due_time, center_y, score))

                    fire_and_log(time.perf_counter())

                    if scheduled:
                        scene_lock_until = now + scene_lock_sec
                        logger.debug(
                            "候选入队: lanes=%s due=%s y=%s score=%s",
                            [_LANES[i] for i, _, _, _ in scheduled],
                            ["%.3f" % due_time for _, due_time, _, _ in scheduled],
                            ["%.1f" % center_y for _, _, center_y, _ in scheduled],
                            ["%.3f" % score for _, _, _, score in scheduled],
                        )
                    elif debug_score_interval and frame_count % debug_score_interval == 0:
                        logger.debug(
                            "鼓面检测分数: d=%.3f f=%.3f j=%.3f k=%.3f candidates=%s",
                            scores[0], scores[1], scores[2], scores[3],
                            [
                                ["%.1f/%.3f" % (c.center_y, c.score) for c in candidates_by_lane[i]]
                                for i in range(_LANE_COUNT)
                            ],
                        )

                if now >= scene_lock_until:
                    gate_state, _ = scene_gate.step(frame)
                    if gate_state != STATE_PLAYING:
                        logger.info(
                            "演奏结束 (帧#%d, 耗时%.1f秒, 冷却锁过期，场景识别=%s)",
                            frame_count, elapsed_total, gate_state,
                        )
                        return CustomAction.RunResult(success=True)
                    if not _is_still_playing(frame, playing_tpl):
                        logger.info(
                            "演奏结束 (帧#%d, 耗时%.1f秒, 冷却锁过期，识别为非playing)",
                            frame_count, elapsed_total,
                        )
                        return CustomAction.RunResult(success=True)
                    scene_lock_until = now + scene_lock_sec

                elapsed_frame = time.perf_counter() - t0
                if elapsed_frame < frame_interval:
                    time.sleep(frame_interval - elapsed_frame)
                now = time.perf_counter()
                key_scheduler.release_expired(now)
                fire_and_log(now)
        finally:
            key_scheduler.close()
