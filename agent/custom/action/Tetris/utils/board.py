import numpy as np

from .pieces import PIECES

BOARD_REGION = [472, 50, 296, 587]
BOARD_COLS = 10
BOARD_ROWS = 20
GRID_LEFT = 17
GRID_TOP = 23
GRID_RIGHT = 301
GRID_BOTTOM = 582
CELL_WIDTH = (GRID_RIGHT - GRID_LEFT) / BOARD_COLS
CELL_HEIGHT = (GRID_BOTTOM - GRID_TOP) / BOARD_ROWS

QUEUE_REGION = [782, 85, 64, 389]


def calculate_column_heights(board: np.ndarray):
    heights = []
    for col in range(BOARD_COLS):
        filled_rows = np.where(board[:, col])[0]
        if len(filled_rows) == 0:
            heights.append(0)
            continue
        heights.append(BOARD_ROWS - int(filled_rows[0]))
    return heights


def calculate_holes(board: np.ndarray):
    holes = 0
    hole_depth = 0
    covered_holes = 0
    hard_holes = 0

    for col in range(BOARD_COLS):
        seen_block = False
        blocks_above = 0
        for row in range(BOARD_ROWS):
            if board[row, col]:
                seen_block = True
                blocks_above += 1
                continue
            if seen_block:
                holes += 1
                hole_depth += BOARD_ROWS - row
                covered_holes += blocks_above
                if blocks_above >= 3:
                    hard_holes += 1

    return holes, hole_depth, covered_holes, hard_holes


def calculate_transitions(board: np.ndarray):
    row_transitions = 0
    for row in range(BOARD_ROWS):
        prev_filled = True
        for col in range(BOARD_COLS):
            filled = bool(board[row, col])
            if filled != prev_filled:
                row_transitions += 1
            prev_filled = filled
        if not prev_filled:
            row_transitions += 1

    col_transitions = 0
    for col in range(BOARD_COLS):
        prev_filled = True
        for row in range(BOARD_ROWS):
            filled = bool(board[row, col])
            if filled != prev_filled:
                col_transitions += 1
            prev_filled = filled
        if not prev_filled:
            col_transitions += 1

    return row_transitions, col_transitions


def calculate_well_penalty(heights: list[int]):
    well_score = 0.0
    for col in range(BOARD_COLS):
        left_height = heights[col - 1] if col > 0 else BOARD_ROWS
        right_height = heights[col + 1] if col < BOARD_COLS - 1 else BOARD_ROWS
        well_depth = max(0, min(left_height, right_height) - heights[col])
        if well_depth > 1:
            well_score += well_depth * (well_depth + 1) / 2
    return well_score


def calculate_open_well_reward(heights: list[int]):
    reward = 0.0
    for col in range(BOARD_COLS):
        left_height = heights[col - 1] if col > 0 else BOARD_ROWS
        right_height = heights[col + 1] if col < BOARD_COLS - 1 else BOARD_ROWS
        well_depth = max(0, min(left_height, right_height) - heights[col])
        if well_depth >= 2:
            reward += well_depth * well_depth
    return reward


def calculate_edge_height_penalty(heights: list[int]):
    if not heights:
        return 0.0
    return float(heights[0] * heights[0] + heights[-1] * heights[-1])


def calculate_center_stack_penalty(heights: list[int]):
    center_cols = heights[3:7]
    avg_center = sum(center_cols) / 4.0
    edge_avg = (
        (sum(heights[:3]) + sum(heights[7:])) / 6.0 if len(heights) >= 10 else 0.0
    )
    surplus = max(0.0, avg_center - edge_avg)
    if surplus <= 2:
        return 0.0
    return float(surplus * surplus)


def calculate_horizontal_balance_penalty(heights: list[int]):
    import math

    n = len(heights)
    if n == 0 or all(h == 0 for h in heights):
        return 0.0
    mean = sum(heights) / n
    variance = sum((h - mean) ** 2 for h in heights) / n
    return float(math.sqrt(variance))


def detect_t_spin(
    board: np.ndarray,
    piece_name: str,
    rotation: int,
    target_col: int,
    drop_row: int,
    was_rotation_move: bool = True,
) -> dict:
    if piece_name != "T":
        return {"is_t_spin": False, "is_mini": False}

    t_shape = PIECES["T"][rotation]

    front_corner_offsets = {
        0: [(-1, -1), (-1, 1)],
        1: [(-1, 1), (1, 1)],
        2: [(1, -1), (1, 1)],
        3: [(-1, -1), (1, -1)],
    }
    back_corner_offsets = {
        0: [(1, -1), (1, 1)],
        1: [(-1, -1), (1, -1)],
        2: [(-1, -1), (-1, 1)],
        3: [(-1, 1), (1, 1)],
    }

    front_corners_blocked = 0
    for row_offset, col_offset in front_corner_offsets[rotation]:
        r = drop_row + row_offset
        c = target_col + col_offset
        if (
            r < 0
            or r >= BOARD_ROWS
            or c < 0
            or c >= BOARD_COLS
            or (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS and board[r, c])
        ):
            front_corners_blocked += 1

    back_corners_blocked = 0
    for row_offset, col_offset in back_corner_offsets[rotation]:
        r = drop_row + row_offset
        c = target_col + col_offset
        if (
            r < 0
            or r >= BOARD_ROWS
            or c < 0
            or c >= BOARD_COLS
            or (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS and board[r, c])
        ):
            back_corners_blocked += 1

    if was_rotation_move:
        is_t_spin = (
            front_corners_blocked >= 2
            and (front_corners_blocked + back_corners_blocked) >= 3
        )
        is_mini = not is_t_spin and front_corners_blocked >= 2
    else:
        is_t_spin = False
        is_mini = (front_corners_blocked + back_corners_blocked) >= 3

    return {"is_t_spin": is_t_spin, "is_mini": is_mini}


def evaluate_board(
    board: np.ndarray,
    lines_cleared: int,
    dynamic_weights: bool = True,
    combo_count: int = 0,
    is_t_spin: bool = False,
):
    heights = calculate_column_heights(board)
    holes, hole_depth, covered_holes, hard_holes = calculate_holes(board)
    row_transitions, col_transitions = calculate_transitions(board)
    well_penalty = calculate_well_penalty(heights)
    open_well_reward = calculate_open_well_reward(heights)
    center_stack_penalty = calculate_center_stack_penalty(heights)
    horizontal_balance_penalty = calculate_horizontal_balance_penalty(heights)

    aggregate_height = sum(heights)
    bumpiness = sum(
        abs(heights[idx] - heights[idx + 1]) for idx in range(len(heights) - 1)
    )

    if dynamic_weights:
        occupancy = np.count_nonzero(board) / (BOARD_ROWS * BOARD_COLS)
        avg_height = aggregate_height / BOARD_COLS

        if avg_height < 8:
            lines_weight = 42.0
            holes_weight = 32.0
            height_weight = 0.95
            bumpiness_weight = 1.4
            transitions_weight = 2.5
            well_weight = 2.8
            open_well_weight = 1.5
            center_stack_weight = 0.5
            balance_weight = 0.3
        elif avg_height < 14:
            lines_weight = 34.18
            holes_weight = 38.99
            height_weight = 1.30
            bumpiness_weight = 1.84
            transitions_weight = 3.21
            well_weight = 3.38
            open_well_weight = 1.0
            center_stack_weight = 1.5
            balance_weight = 0.8
        else:
            lines_weight = 28.0
            holes_weight = 52.0
            height_weight = 1.85
            bumpiness_weight = 2.2
            transitions_weight = 4.0
            well_weight = 4.5
            open_well_weight = 0.5
            center_stack_weight = 3.0
            balance_weight = 1.5

        if occupancy > 0.45:
            holes_weight *= 1.3
            well_weight *= 1.2
    else:
        lines_weight = 34.18
        holes_weight = 38.99
        height_weight = 1.30
        bumpiness_weight = 1.84
        transitions_weight = 3.21
        well_weight = 3.38
        open_well_weight = 1.0
        center_stack_weight = 1.5
        balance_weight = 0.8

    score = (
        lines_cleared * lines_weight
        - aggregate_height * height_weight
        - holes * holes_weight
        - hard_holes * (holes_weight * 0.5)
        - bumpiness * bumpiness_weight
        - row_transitions * transitions_weight
        - col_transitions * (transitions_weight * 2.9)
        - well_penalty * well_weight
        + open_well_reward * open_well_weight
        - center_stack_penalty * center_stack_weight
        - horizontal_balance_penalty * balance_weight
    )

    if combo_count > 1:
        score += combo_count * 25.0

    if is_t_spin:
        t_spin_lines_bonus = lines_cleared * 30.0
        score += t_spin_lines_bonus + 40.0

    return score


def evaluate_board_fast(board: np.ndarray, lines_cleared: int, combo_count: int = 0):
    heights = calculate_column_heights(board)
    holes, _, _, hard_holes = calculate_holes(board)
    aggregate_height = sum(heights)
    bumpiness = sum(
        abs(heights[idx] - heights[idx + 1]) for idx in range(len(heights) - 1)
    )
    center_stack_penalty = calculate_center_stack_penalty(heights)
    balance_penalty = calculate_horizontal_balance_penalty(heights)

    score = (
        lines_cleared * 35.0
        - aggregate_height * 1.2
        - holes * 40.0
        - hard_holes * 20.0
        - bumpiness * 1.5
        - center_stack_penalty * 1.5
        - balance_penalty * 0.8
    )

    if combo_count > 1:
        score += combo_count * 25.0

    return score


def simulate_drop(board: np.ndarray, shape, target_col: int):
    shape_height = max(row for row, _ in shape) + 1
    shape_width = max(col for _, col in shape) + 1
    if target_col < 0 or target_col + shape_width > BOARD_COLS:
        return None

    def collides(test_row: int):
        for row_offset, col_offset in shape:
            row = test_row + row_offset
            col = target_col + col_offset
            if row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
                return True
            if row >= 0 and board[row, col]:
                return True
        return False

    drop_row = 0
    if collides(drop_row):
        return None

    while not collides(drop_row + 1):
        drop_row += 1

    new_board = board.copy()
    for row_offset, col_offset in shape:
        new_board[drop_row + row_offset, target_col + col_offset] = True

    full_rows = [row for row in range(BOARD_ROWS) if np.all(new_board[row])]
    lines_cleared = len(full_rows)
    if lines_cleared:
        new_board = np.delete(new_board, full_rows, axis=0)
        new_board = np.vstack(
            [np.zeros((lines_cleared, BOARD_COLS), dtype=bool), new_board]
        )

    landing_height = BOARD_ROWS - (drop_row + (shape_height / 2.0))
    score = evaluate_board(new_board, lines_cleared)
    score -= landing_height * 4.5

    piece_name = None
    for name, rotations in PIECES.items():
        for rot_shape in rotations:
            if tuple(sorted(rot_shape)) == tuple(sorted(shape)):
                piece_name = name
                break
        if piece_name:
            break

    is_t_spin = False
    is_mini = False
    if piece_name == "T":
        rotation = None
        for rot_idx, rot_shape in enumerate(PIECES["T"]):
            if tuple(sorted(rot_shape)) == tuple(sorted(shape)):
                rotation = rot_idx
                break
        if rotation is not None:
            t_spin_result = detect_t_spin(
                board, "T", rotation, target_col, drop_row, was_rotation_move=False
            )
            is_t_spin = t_spin_result["is_t_spin"]
            is_mini = t_spin_result["is_mini"]

    landing_info = {
        "combo_lines": lines_cleared,
        "is_t_spin": is_t_spin,
        "is_mini": is_mini,
        "cleared_rows": full_rows,
        "drop_row": drop_row,
    }

    return {
        "score": score,
        "lines_cleared": lines_cleared,
        "row": drop_row,
        "board": new_board,
        "is_t_spin": is_t_spin,
        "landing_info": landing_info,
    }


def extract_board_crop(img):
    import cv2

    x, y, w, h = BOARD_REGION
    crop = img[y : y + h, x : x + w]
    if len(crop.shape) == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    return crop


def extract_queue_crop(img):
    import cv2

    x, y, w, h = QUEUE_REGION
    crop = img[y : y + h, x : x + w]
    if len(crop.shape) == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    return crop
