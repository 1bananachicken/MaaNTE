import ctypes
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from PIL import Image

import cv2
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import Print


_KEY_LABELS = {
    0: "none",
    1: "A",
    2: "D",
    3: "W",
    4: "S",
    5: "AW",
    6: "AS",
    7: "DW",
    8: "DS",
}
_VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44}
_EXAMPLES_PER_SECOND = 2.0
_SEQUENCE_LENGTH = 5
_IMAGE_SIZE = (480, 270)
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "debug" / "dataset"


def _parse_params(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            logger.warning(f"invalid dataset_recorder params: {raw!r}")
    return {}


def _resolve_output_dir(value: Any) -> Path:
    if not value:
        return _DEFAULT_OUTPUT_DIR
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _make_session_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / timestamp
    suffix = 1
    while session_dir.exists():
        session_dir = base_dir / f"{timestamp}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _pressed_keys() -> set[str]:
    pressed = set()
    for key, vk in _VK.items():
        if ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000:
            pressed.add(key)
    return pressed


def _label_from_keys(pressed: set[str]) -> int:
    if pressed == {"A"}:
        return 1
    if pressed == {"D"}:
        return 2
    if pressed == {"W"}:
        return 3
    if pressed == {"S"}:
        return 4
    if pressed == {"A", "W"}:
        return 5
    if pressed == {"A", "S"}:
        return 6
    if pressed == {"D", "W"}:
        return 7
    if pressed == {"D", "S"}:
        return 8
    return 0


def _prepare_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.resize(frame, size)


def _save_sample(
    output_dir: Path,
    frames: list[np.ndarray],
    labels: list[int],
    number: int,
) -> Path:
    label_part = "_".join(str(label) for label in labels)
    filename = f"K{number}%{label_part}.jpeg"
    path = output_dir / filename
    image = np.concatenate(frames, axis=1)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
    return path


@AgentServer.custom_action("autonomous_driving_dataset_recorder")
class AutonomousDrivingDatasetRecorder(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = _parse_params(argv.custom_action_param)
        dataset_dir = _resolve_output_dir(params.get("output_dir"))
        print(f"Dataset recorder output directory: {dataset_dir}")
        output_dir = _make_session_dir(dataset_dir)
        print(f"Dataset recorder session directory: {output_dir}")

        try:
            duration_seconds = max(0.0, float(params.get("duration_seconds", 60.0)))
        except (TypeError, ValueError):
            duration_seconds = 60.0
        try:
            start_delay_seconds = max(
                0.0, float(params.get("start_delay_seconds", 1.0))
            )
        except (TypeError, ValueError):
            start_delay_seconds = 1.0

        metadata = {
            "format": "K<number>%<label>_<label>_...jpeg",
            "labels": _KEY_LABELS,
            "sequence_length": _SEQUENCE_LENGTH,
            "image_width": _IMAGE_SIZE[0],
            "image_height": _IMAGE_SIZE[1],
            "examples_per_second": _EXAMPLES_PER_SECOND,
            "start_delay_seconds": start_delay_seconds,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )

        controller = context.tasker.controller
        frames = [
            np.zeros((_IMAGE_SIZE[1], _IMAGE_SIZE[0], 3), dtype=np.uint8)
            for _ in range(_SEQUENCE_LENGTH)
        ]
        labels = [0 for _ in range(_SEQUENCE_LENGTH)]
        sample_no = 0
        saved_count = 0
        captured_count = 0
        deadline = time.time() + duration_seconds if duration_seconds > 0 else None
        last_status = 0.0

        Print(
            context,
            f"Dataset recorder started: {output_dir} "
            f"({_EXAMPLES_PER_SECOND:g} samples/s, {duration_seconds:g}s, "
            f"delay={start_delay_seconds:g}s)",
        )

        delay_deadline = time.time() + start_delay_seconds
        while not context.tasker.stopping and time.time() < delay_deadline:
            remaining = max(0.0, delay_deadline - time.time())
            Print(context, f"Dataset recorder starts in {remaining:.1f}s")
            time.sleep(min(1.0, remaining))

        while not context.tasker.stopping:
            if deadline is not None and time.time() >= deadline:
                break
            start = time.time()
            image = controller.post_screencap().wait().get()
            frame = _prepare_frame(image, _IMAGE_SIZE)
            if frame is None:
                logger.warning("dataset_recorder: empty screenshot, retrying")
                time.sleep(0.1)
                continue

            pressed = _pressed_keys()
            label = _label_from_keys(pressed)
            frames = frames[1:] + [frame]
            labels = labels[1:] + [label]
            captured_count += 1

            if captured_count < _SEQUENCE_LENGTH:
                wait_time = (start + 1.0 / _EXAMPLES_PER_SECOND) - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)
                continue

            _save_sample(output_dir, frames, labels, sample_no)
            sample_no += 1
            saved_count += 1

            now = time.time()
            if now - last_status >= 2.0:
                Print(
                    context,
                    f"Dataset recorder: saved={saved_count}, "
                    f"last={_KEY_LABELS[label]}, keys={''.join(sorted(pressed)) or '-'}",
                )
                last_status = now

            wait_time = (start + 1.0 / _EXAMPLES_PER_SECOND) - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

        Print(context, f"Dataset recorder stopped: saved={saved_count}, dir={output_dir}")
        return CustomAction.RunResult(success=True)
