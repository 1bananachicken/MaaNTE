PIECES = {
    "I": [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ],
    "O": [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ],
    "T": [
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ],
    "S": [
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ],
    "Z": [
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ],
    "J": [
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ],
    "L": [
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ],
}


def normalize_cells(cells):
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    normalized = tuple(sorted((row - min_row, col - min_col) for row, col in cells))
    return normalized, min_row, min_col


def match_piece_state(cells):
    normalized, min_row, min_col = normalize_cells(cells)
    for piece_name, rotations in PIECES.items():
        for rotation_index, shape in enumerate(rotations):
            if normalized == tuple(sorted(shape)):
                return {
                    "piece": piece_name,
                    "rotation": rotation_index,
                    "row": min_row,
                    "col": min_col,
                    "cells": tuple(sorted(cells)),
                }
    return None


def rotation_distance(piece_name: str, current_rotation: int, target_rotation: int):
    rotation_count = len(PIECES[piece_name])
    clockwise_steps = (target_rotation - current_rotation) % rotation_count
    counterclockwise_steps = (current_rotation - target_rotation) % rotation_count
    return min(clockwise_steps, counterclockwise_steps)
