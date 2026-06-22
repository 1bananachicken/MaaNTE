#!/usr/bin/env python3
"""Fit local Navi calibration points and update runtime transform constants."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "config" / "navi_coordinate_calibration.json"
DEFAULT_TARGET = (
    PROJECT_ROOT / "agent" / "custom" / "action" / "Navi" / "coordinate_position.py"
)
MAP_WORLD_ORIGIN = (5632.0, 5632.0)
MAP_PIXELS_PER_WORLD_UNIT = 22.0
PLANE_AXES = ((0, 1), (0, 2), (1, 2))
BEGIN_MARKER = "# BEGIN GENERATED NAVI COORDINATE TRANSFORM"
END_MARKER = "# END GENERATED NAVI COORDINATE TRANSFORM"


def parse_point(value: Any, index: int) -> tuple[tuple[float, float, float], tuple[float, float]]:
    if not isinstance(value, dict):
        raise ValueError(f"point {index} must be an object")
    raw = value.get("raw")
    map_point = value.get("map")
    world = value.get("world")
    if map_point is None and isinstance(world, (list, tuple)) and len(world) == 2:
        latitude = float(world[0])
        longitude = float(world[1])
        map_point = (
            MAP_WORLD_ORIGIN[0] + longitude * MAP_PIXELS_PER_WORLD_UNIT,
            MAP_WORLD_ORIGIN[1] - latitude * MAP_PIXELS_PER_WORLD_UNIT,
        )
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"point {index} needs raw=[x,y,z]")
    if not isinstance(map_point, (list, tuple)) or len(map_point) != 2:
        raise ValueError(f"point {index} needs map=[x,y] or world=[lat,lng]")
    return (
        (float(raw[0]), float(raw[1]), float(raw[2])),
        (float(map_point[0]), float(map_point[1])),
    )


def fit_transform(
    samples: list[tuple[tuple[float, float, float], tuple[float, float]]],
) -> tuple[tuple[int, int], float, float, float, float, float, list[float]]:
    if len(samples) < 3:
        raise ValueError("at least 3 calibration points are required")

    candidates = []
    for axes in PLANE_AXES:
        matrix = []
        target = []
        for raw, point in samples:
            raw_x = raw[axes[0]]
            raw_y = raw[axes[1]]
            map_x, map_y = point
            matrix.append([raw_x, -raw_y, 1.0, 0.0])
            target.append(map_x)
            matrix.append([raw_y, raw_x, 0.0, 1.0])
            target.append(map_y)

        coefficients, _, rank, _ = np.linalg.lstsq(
            np.asarray(matrix, dtype=np.float64),
            np.asarray(target, dtype=np.float64),
            rcond=None,
        )
        if rank < 4:
            continue
        a, b, tx, ty = (float(value) for value in coefficients)
        scale = math.hypot(a, b)
        if not 0.0001 <= scale <= 100.0:
            continue

        errors = []
        for raw, point in samples:
            raw_x = raw[axes[0]]
            raw_y = raw[axes[1]]
            predicted = (
                a * raw_x - b * raw_y + tx,
                b * raw_x + a * raw_y + ty,
            )
            errors.append(math.dist(predicted, point))
        rms = math.sqrt(sum(error * error for error in errors) / len(errors))
        candidates.append((axes, a, b, tx, ty, rms, errors))

    if not candidates:
        raise ValueError("calibration points cannot produce a valid transform")
    return min(candidates, key=lambda item: item[5])


def render_constants(
    axes: tuple[int, int],
    a: float,
    b: float,
    tx: float,
    ty: float,
    error: float,
) -> str:
    return "\n".join(
        (
            BEGIN_MARKER,
            f"_CALIBRATION_AXES = {axes!r}",
            f"_CALIBRATION_A = {a!r}",
            f"_CALIBRATION_B = {b!r}",
            f"_CALIBRATION_TX = {tx!r}",
            f"_CALIBRATION_TY = {ty!r}",
            f"_CALIBRATION_ERROR = {error!r}",
            END_MARKER,
        )
    )


def update_target(path: Path, constants: str) -> None:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )
    updated, count = pattern.subn(constants, source, count=1)
    if count != 1:
        raise ValueError(f"generated transform markers not found in {path}")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Navi calibration points and update runtime constants."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--max-error",
        type=float,
        default=80.0,
        help="Maximum accepted RMS error in map pixels.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Calculate and print parameters without updating the target.",
    )
    args = parser.parse_args()

    value = json.loads(args.input.read_text(encoding="utf-8"))
    points = value.get("points")
    if not isinstance(points, list):
        raise ValueError("calibration file must contain a points array")
    samples = [parse_point(item, index) for index, item in enumerate(points)]
    axes, a, b, tx, ty, error, errors = fit_transform(samples)
    if error > args.max_error:
        raise ValueError(
            f"calibration RMS error {error:.3f} exceeds limit {args.max_error:.3f}"
        )

    constants = render_constants(axes, a, b, tx, ty, error)
    print(constants)
    print(f"scale={math.hypot(a, b):.12f}")
    print("point_errors=" + ", ".join(f"{item:.3f}" for item in errors))
    if not args.check:
        update_target(args.target, constants)
        print(f"updated={args.target}")


if __name__ == "__main__":
    main()
