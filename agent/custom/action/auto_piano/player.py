from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from maa.context import Context

from .maa_keyboard import MaaKeyboardBridge
from .midi_processor import MidiProcessor

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

        parsed = self.processor.parse(str(song_path))
        notes = parsed["notes"]
        if not notes:
            print(f"No playable notes found: {song_path}", flush=True)
            return 0

        print(
            f"AutoPiano loaded {parsed['title']} "
            f"({len(notes)} notes, speed {settings.speed}x, transpose {settings.transpose})",
            flush=True,
        )

        self.sleep_interruptibly(context, DEFAULT_COUNTDOWN)
        bridge = MaaKeyboardBridge(self.get_controller(context))
        played = self.play_notes(
            context, notes, bridge, settings.speed, settings.transpose
        )
        print(
            f"AutoPiano finished. {played} notes sent through Maa controller.",
            flush=True,
        )
        return played

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

    @staticmethod
    def adjust_pitch(midi_pitch: int) -> int:
        while midi_pitch < 60:
            midi_pitch += 12
        while midi_pitch > 95:
            midi_pitch -= 12
        return midi_pitch

    @classmethod
    def collect_chord(
        cls,
        notes: list[dict],
        start_index: int,
        transpose: int,
        window: float,
    ) -> tuple[list[int], int]:
        base_time = notes[start_index]["t"]
        chord = []
        index = start_index

        while index < len(notes) and abs(notes[index]["t"] - base_time) <= window:
            pitch = int(notes[index]["p"]) + transpose
            chord.append(cls.adjust_pitch(pitch))
            index += 1

        return chord, index

    @classmethod
    def play_notes(
        cls,
        context: Context,
        notes: list[dict],
        bridge: MaaKeyboardBridge,
        speed: float,
        transpose: int,
    ) -> int:
        start_time = time.perf_counter()
        elapsed = 0.0
        index = 0
        played = 0

        while index < len(notes):
            cls.raise_if_stopped(context)
            target_time = float(notes[index]["t"]) / speed
            chord, next_index = cls.collect_chord(
                notes, index, transpose, DEFAULT_CHORD_WINDOW
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
