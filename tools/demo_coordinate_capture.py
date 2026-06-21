from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRDPARTY_DIR = PROJECT_ROOT / "thirdparty"
MODULE_NAME = "nte_coordinate_api"
CALIBRATION_AXES = (0, 1)
CALIBRATION_A = 0.016394586684750773
CALIBRATION_B = 5.693519256055879e-08
CALIBRATION_TX = 6293.474380746091
CALIBRATION_TY = 3472.664390686138


def _ensure_project_paths() -> None:
    for path in (THIRDPARTY_DIR,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_coordinate_module() -> Any:
    _ensure_project_paths()
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError:
        pass

    candidates = sorted(THIRDPARTY_DIR.glob("%s*.pyd" % MODULE_NAME))
    if not candidates:
        raise FileNotFoundError(
            "No coordinate module found under %s" % THIRDPARTY_DIR
        )

    last_error: Exception | None = None
    for candidate in candidates:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            if sys.modules.get(MODULE_NAME) is module:
                sys.modules.pop(MODULE_NAME, None)
            last_error = exc
            continue
        setattr(module, "_coordinate_demo_origin", str(candidate))
        return module

    if last_error is not None:
        raise last_error
    raise ImportError("Unable to load %s from %s" % (MODULE_NAME, THIRDPARTY_DIR))


def map_point_from_raw(raw: tuple[float, float, float]) -> tuple[int, int] | None:
    x = raw[CALIBRATION_AXES[0]]
    y = raw[CALIBRATION_AXES[1]]
    map_x = CALIBRATION_A * x - CALIBRATION_B * y + CALIBRATION_TX
    map_y = CALIBRATION_B * x + CALIBRATION_A * y + CALIBRATION_TY
    if not all(value == value for value in (map_x, map_y)):
        return None
    return int(round(map_x)), int(round(map_y))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: read the current character coordinate from nte_coordinate_api."
    )
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-age", type=float, default=2.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after the first valid coordinate sample.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    module = load_coordinate_module()
    capture_type = getattr(module, "CoordinateCapture")
    capture = capture_type()

    origin = getattr(module, "_coordinate_demo_origin", None)
    if origin is None:
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
    print("loaded module: %s" % (origin or getattr(module, "__file__", "<unknown>")))
    print(
        "sampling for %.1fs, interval=%.2fs, max_age=%.2fs"
        % (args.seconds, args.interval, args.max_age)
    )

    start = time.monotonic()
    samples = 0
    try:
        try:
            capture.start()
        except Exception as exc:
            print(
                "failed to start coordinate capture: %s: %s"
                % (type(exc).__name__, exc)
            )
            return 1
        deadline = start + max(args.seconds, 0.0)
        while time.monotonic() <= deadline:
            raw = capture.read(max_age=args.max_age)
            elapsed = time.monotonic() - start
            if raw is None:
                print("[%.2fs] no coordinate sample" % elapsed)
            else:
                samples += 1
                raw_tuple = tuple(float(value) for value in raw)
                map_point = map_point_from_raw(raw_tuple)
                if map_point is None:
                    print(
                        "[%.2fs] raw=(%.3f, %.3f, %.3f)"
                        % (elapsed, raw_tuple[0], raw_tuple[1], raw_tuple[2])
                    )
                else:
                    print(
                        "[%.2fs] raw=(%.3f, %.3f, %.3f) map=(%d, %d)"
                        % (
                            elapsed,
                            raw_tuple[0],
                            raw_tuple[1],
                            raw_tuple[2],
                            map_point[0],
                            map_point[1],
                        )
                    )
                if args.once:
                    return 0
            time.sleep(max(args.interval, 0.05))
    finally:
        try:
            capture.close()
        except Exception as exc:
            print(
                "coordinate capture close warning: %s: %s"
                % (type(exc).__name__, exc)
            )

    if samples == 0:
        print("no valid coordinate captured")
        return 2
    print("captured %d coordinate sample(s)" % samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
