from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from maa.context import Context

from utils.maafocus import PrintT
from .maa_keyboard import MaaKeyboardBridge
from .midi_processor import MidiProcessor
from . import key_mapping

DEFAULT_SPEED = 1.0
DEFAULT_TRANSPOSE = 0
DEFAULT_COUNTDOWN = 1

# 定时精度阈值：大于此值用 sleep，小于此值忙等待（保证精度同时降低 CPU）
_WAIT_THRESHOLD = 0.003


class AutoPianoStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class AutoPianoSettings:
    song: str
    speed: float = DEFAULT_SPEED
    transpose: int = DEFAULT_TRANSPOSE
    key_mode: str = "36"
    tracks: str = "all"  # "all" | "melody" | 轨道索引列表由 action 处理
    out_of_range_mode: str = "fold"  # "fold" | "shift" | "cut"
    sustain: bool = True  # True = 按 MIDI 时长延音; False = 短触键（调试模式）
    sustain_mode: str = "repeat"  # "repeat" = 高频补发; "hold" = 按住到音符结束
    sustain_repeat_rate: str = "fast"  # slow | normal | fast | turbo


class AutoPianoPlayer:
    def __init__(self, project_root: Path, processor: MidiProcessor | None = None):
        self.project_root = project_root
        self.processor = processor or MidiProcessor()

    def play(self, context: Context, settings: AutoPianoSettings) -> int:
        if not settings.song:
            raise ValueError("custom_action_param.song is required.")
        if settings.speed <= 0:
            raise ValueError("speed must be greater than 0.")

        song_path = self.resolve_song_path(settings.song)
        if not song_path.is_file():
            raise FileNotFoundError(f"Song file not found: {song_path}")

        # 解析轨道选择参数
        tracks_param = self._parse_tracks_param(settings.tracks)

        parsed = self.processor.parse(str(song_path), tracks=tracks_param)
        notes = parsed["notes"]
        if not notes:
            PrintT(context, "auto_piano.no_notes", str(song_path))
            return 0

        # 根据超出音域模式处理音符
        notes, effective_transpose = self._apply_range_mode(notes, settings)
        if not notes:
            PrintT(context, "auto_piano.no_playable_notes")
            return 0

        PrintT(
            context,
            "auto_piano.loaded",
            parsed["title"],
            len(notes),
            settings.speed,
            effective_transpose,
        )
        if parsed.get("track_count", 1) > 1:
            PrintT(
                context,
                "auto_piano.tracks_info",
                parsed["track_count"],
                parsed.get("parsed_tracks", []),
            )

        # 生成按键事件列表
        mapping = key_mapping.get_mapping(settings.key_mode)
        events = self._build_events(
            notes, mapping, settings.speed, effective_transpose, settings.key_mode, settings.out_of_range_mode, settings.sustain
        )
        if not events:
            PrintT(context, "auto_piano.no_playable_notes")
            return 0

        self.sleep_interruptibly(context, DEFAULT_COUNTDOWN)
        sustain_mode = (
            settings.sustain_mode if settings.sustain_mode in ("repeat", "hold") else "repeat"
        )
        repeat_interval = self._parse_repeat_interval(settings.sustain_repeat_rate)
        bridge = MaaKeyboardBridge(
            self.get_controller(context),
            wait_jobs=True,
            mapping=mapping,
            sustain_mode=sustain_mode,
            repeat_interval=repeat_interval,
        )

        try:
            played = self._play_events(context, events, bridge, len(notes))
        finally:
            # 无论成功失败，确保所有键都被释放
            bridge.release_all()

        PrintT(context, "auto_piano.finished", played)
        return played

    @staticmethod
    def _parse_tracks_param(tracks_raw: str) -> str | list[int]:
        """将 tracks 字符串参数转换为解析器需要的格式。"""
        tracks_str = str(tracks_raw).strip().lower()
        if tracks_str in ("all", "melody", ""):
            return tracks_str or "all"
        # 尝试解析为逗号分隔的轨道索引，例如 "0,2"
        try:
            indices = [int(x.strip()) for x in tracks_str.split(",") if x.strip()]
            if indices:
                return indices
        except ValueError:
            pass
        return "all"

    @staticmethod
    def _parse_repeat_interval(rate_raw: str) -> float:
        rate = str(rate_raw).strip().lower()
        rate_to_interval = {
            "slow": 0.09,
            "normal": 0.06,
            "fast": 0.045,
            "turbo": 0.03,
        }
        return rate_to_interval.get(rate, rate_to_interval["fast"])

    def resolve_song_path(self, song: str) -> Path:
        path = Path(song).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    @staticmethod
    def get_controller(context: Context):
        controller = getattr(context, "controller", None)
        if controller is not None:
            return controller

        tasker = getattr(context, "tasker", None)
        controller = getattr(tasker, "controller", None)
        if controller is not None:
            return controller

        raise RuntimeError("Maa controller is unavailable in custom action context.")

    @staticmethod
    def is_stop_requested(context: Context) -> bool:
        tasker = getattr(context, "tasker", None)
        if tasker is None:
            return False

        stopping = getattr(tasker, "stopping", False)
        if callable(stopping):
            stopping = stopping()
        return bool(stopping)

    @classmethod
    def raise_if_stopped(cls, context: Context) -> None:
        if cls.is_stop_requested(context):
            raise AutoPianoStopped("Auto piano playback stopped by Maa tasker.")

    @classmethod
    def sleep_interruptibly(
        cls, context: Context, seconds: float, step: float = 0.02
    ) -> None:
        deadline = time.perf_counter() + seconds

        while True:
            cls.raise_if_stopped(context)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(step, remaining))

    @classmethod
    def _apply_range_mode(
        cls, notes: list[dict], settings: AutoPianoSettings
    ) -> tuple[list[dict], int]:
        """
        根据超出音域处理模式预处理音符。
        返回: (处理后的音符列表, 有效移调值)
        """
        mode = settings.out_of_range_mode
        transpose = settings.transpose

        if mode == "fold":
            return notes, transpose

        if mode == "shift":
            pitches = [n["p"] for n in notes]
            center = (min(pitches) + max(pitches)) / 2
            # 目标中心: 77.5 = (60 + 95) / 2
            auto_shift = round(77.5 - center)
            effective_transpose = transpose + auto_shift
            filtered = []
            for note in notes:
                p = note["p"] + effective_transpose
                if 60 <= p <= 95:
                    filtered.append({**note, "p": p})
            return filtered, effective_transpose

        if mode == "cut":
            effective_transpose = transpose
            filtered = []
            for note in notes:
                p = note["p"] + effective_transpose
                if 60 <= p <= 95:
                    filtered.append({**note, "p": p})
            return filtered, effective_transpose

        # 未知模式回退到 fold
        return notes, transpose

    @classmethod
    def adjust_pitch(cls, midi_pitch: int, key_mode: str = "36") -> int | None:
        """调整音高到可用范围，并过滤掉无法映射的音符。"""
        # 21键模式：先将半音映射到最近白键
        if key_mode == "21":
            midi_pitch = key_mapping.snap_to_white_key(midi_pitch)

        # 折叠到游戏钢琴支持的 60~95 范围
        while midi_pitch < 60:
            midi_pitch += 12
        while midi_pitch > 95:
            midi_pitch -= 12
        return midi_pitch

    # ------------------------------------------------------------------
    # 事件生成
    # ------------------------------------------------------------------
    # 非延音模式下的固定触键时长（秒）
    _STACCATO_DURATION = 0.05
    _MIN_NOTE_DURATION = 0.01

    @classmethod
    def _build_events(
        cls,
        notes: list[dict],
        mapping: dict[int, str],
        speed: float,
        transpose: int,
        key_mode: str,
        range_mode: str = "fold",
        sustain: bool = True,
    ) -> list[tuple[float, int, bool]]:
        """
        将音符列表转换为按键事件列表。
        每个事件: (时间点, midi_pitch, is_press)
        """
        events = []
        for note in notes:
            raw_pitch = int(note["p"])
            if range_mode == "fold":
                raw_pitch += transpose
                pitch = cls.adjust_pitch(raw_pitch, key_mode)
            else:
                # shift / cut 模式下，音符已在 _apply_range_mode 中处理并过滤
                pitch = raw_pitch
                if key_mode == "21":
                    pitch = key_mapping.snap_to_white_key(pitch)

            if pitch is None or pitch not in mapping:
                continue

            start = float(note["t"]) / speed
            if sustain:
                duration = max(float(note.get("d", 0.05)) / speed, cls._MIN_NOTE_DURATION)
            else:
                # 非延音模式：固定短触键，方便调试
                duration = max(cls._STACCATO_DURATION / speed, cls._MIN_NOTE_DURATION)
            end = start + duration

            events.append((start, pitch, True))   # press
            events.append((end, pitch, False))    # release

        # 排序：时间升序；同一时刻，release (False=0) 排在 press (True=1) 前面
        events.sort(key=lambda e: (e[0], e[2]))
        return events

    # ------------------------------------------------------------------
    # 事件播放（核心）
    # ------------------------------------------------------------------
    @classmethod
    def _play_events(
        cls,
        context: Context,
        events: list[tuple[float, int, bool]],
        bridge: MaaKeyboardBridge,
        total_notes: int,
    ) -> int:
        start_time = time.perf_counter()
        played = 0
        report_interval = max(1, total_notes // 10)
        next_report = report_interval

        for evt_time, pitch, is_press in events:
            cls.raise_if_stopped(context)
            cls._wait_until(context, start_time + evt_time, bridge)

            if is_press:
                bridge.press_note(pitch)
                played += 1
                if played >= next_report:
                    PrintT(
                        context,
                        "auto_piano.progress",
                        played,
                        total_notes,
                        int(played * 100 / total_notes),
                    )
                    next_report += report_interval
            else:
                bridge.release_note(pitch)

        return played

    # ------------------------------------------------------------------
    # 低 CPU 高精度等待
    # ------------------------------------------------------------------
    @classmethod
    def _wait_until(
        cls, context: Context, deadline: float, bridge: MaaKeyboardBridge | None = None
    ) -> None:
        """
        混合等待策略：
        - 剩余时间较大时，用 time.sleep() 让出 CPU；
        - 长等待分段睡，每 0.1 秒检查一次停止信号；
        - 接近目标时，小范围忙等待保证按键时序精度。
        """
        while True:
            cls.raise_if_stopped(context)
            if bridge is not None:
                bridge.refresh_active_keys()
            now = time.perf_counter()
            remaining = deadline - now
            if remaining <= 0:
                if bridge is not None:
                    bridge.refresh_active_keys(force=True)
                return

            if bridge is not None and getattr(bridge, "_active_counts", None):
                if getattr(bridge, "sustain_mode", "hold") == "repeat":
                    active_step = min(max(bridge.repeat_interval / 2, 0.005), 0.02)
                else:
                    active_step = 0.05
                time.sleep(min(active_step, remaining))
            elif remaining > 0.5:
                # 长等待分段睡，防止 sleep 期间无法响应停止
                time.sleep(0.1)
            elif remaining > _WAIT_THRESHOLD:
                # 提前 _WAIT_THRESHOLD 秒唤醒，再进入忙等待收尾
                time.sleep(remaining - _WAIT_THRESHOLD)
            elif remaining > 0.0003:
                # 最后 0.3ms 用更细粒度 sleep（避免完全忙等待吃满 CPU）
                time.sleep(0.0002)
            # 最后几百微秒自然循环收尾，保证精度
