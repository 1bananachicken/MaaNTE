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
DEFAULT_CHORD_WINDOW = 0.005


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

        # 转换到游戏可用音域，再按原版逻辑分组成短和弦。
        mapping = key_mapping.get_mapping(settings.key_mode)
        playable_notes = self._build_playable_notes(
            notes,
            mapping,
            effective_transpose,
            settings.key_mode,
            settings.out_of_range_mode,
        )
        if not playable_notes:
            PrintT(context, "auto_piano.no_playable_notes")
            return 0

        self.sleep_interruptibly(context, DEFAULT_COUNTDOWN)
        bridge = MaaKeyboardBridge(mapping=mapping)
        played = self.play_notes(
            context,
            playable_notes,
            bridge,
            settings.speed,
        )

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

    def resolve_song_path(self, song: str) -> Path:
        path = Path(song).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

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

    @classmethod
    def _build_playable_notes(
        cls,
        notes: list[dict],
        mapping: dict[int, str],
        transpose: int,
        key_mode: str,
        range_mode: str = "fold",
    ) -> list[dict]:
        """将音符转换到当前键位模式可播放的音高。"""
        playable_notes = []
        for note in notes:
            pitch = int(note["p"])
            if range_mode == "fold":
                pitch = cls.adjust_pitch(pitch + transpose, key_mode)
            elif key_mode == "21":
                pitch = key_mapping.snap_to_white_key(pitch)

            if pitch in mapping:
                playable_notes.append({**note, "p": pitch})

        return playable_notes

    @staticmethod
    def collect_chord(
        notes: list[dict], start_index: int, window: float
    ) -> tuple[list[int], int]:
        """收集指定时间窗口内开始的音符。"""
        base_time = notes[start_index]["t"]
        chord = []
        index = start_index

        while index < len(notes) and abs(notes[index]["t"] - base_time) <= window:
            chord.append(int(notes[index]["p"]))
            index += 1

        return chord, index

    @classmethod
    def play_notes(
        cls,
        context: Context,
        notes: list[dict],
        bridge: MaaKeyboardBridge,
        speed: float,
    ) -> int:
        """按原版节奏将音符分组成短和弦播放。"""
        start_time = time.perf_counter()
        elapsed = 0.0
        index = 0
        played = 0

        while index < len(notes):
            cls.raise_if_stopped(context)
            target_time = float(notes[index]["t"]) / speed
            chord, next_index = cls.collect_chord(
                notes, index, DEFAULT_CHORD_WINDOW
            )

            while elapsed < target_time:
                cls.raise_if_stopped(context)
                elapsed = time.perf_counter() - start_time
                remaining = target_time - elapsed
                if remaining > 0:
                    time.sleep(min(0.002, remaining))

            cls.raise_if_stopped(context)
            bridge.execute_chord(chord)
            elapsed = time.perf_counter() - start_time
            played += len(chord)
            index = next_index

        return played
