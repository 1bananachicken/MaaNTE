from collections import deque

import numpy as np

from .pieces import PIECES

BOARD_REGION = [473, 50, 294, 587]
BOARD_COLS = 10
BOARD_ROWS = 20
GRID_LEFT = 17
GRID_TOP = 23
GRID_RIGHT = 301
GRID_BOTTOM = 582
CELL_WIDTH = (GRID_RIGHT - GRID_LEFT) / BOARD_COLS
CELL_HEIGHT = (GRID_BOTTOM - GRID_TOP) / BOARD_ROWS

REAL_BLOCK_VALUE_THRESHOLD = 160
REAL_BLOCK_SATURATION_THRESHOLD = 80

DEBUG_BOARD = False  # 设为 True 启用棋盘识别调试输出

BOARD_LINE_COLUMNS = [17, 47, 76, 105, 135, 164, 193, 223, 252, 281, 301]
BOARD_LINE_ROWS = [
    23,
    53,
    82,
    111,
    141,
    170,
    199,
    229,
    258,
    287,
    316,
    346,
    375,
    404,
    434,
    463,
    492,
    522,
    551,
    580,
]
BOARD_INTERIOR_COLUMNS = [32, 62, 91, 120, 149, 178, 208, 237, 266, 291]
BOARD_INTERIOR_ROWS = [
    38,
    67,
    96,
    126,
    155,
    184,
    214,
    243,
    272,
    301,
    331,
    360,
    389,
    419,
    448,
    477,
    507,
    536,
    565,
]

QUEUE_REGION = [782, 85, 64, 389]


def collect_components(grid: np.ndarray):
    components = []
    visited = np.zeros_like(grid, dtype=bool)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if not grid[row, col] or visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            component = []

            while queue:
                cur_row, cur_col = queue.popleft()
                component.append((cur_row, cur_col))
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if (
                        0 <= next_row < grid.shape[0]
                        and 0 <= next_col < grid.shape[1]
                        and grid[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))

            components.append(component)

    return components


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

    return holes, hole_depth, covered_holes


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


def calculate_top_occupancy_penalty(board: np.ndarray, rows: int = 4):
    top_rows = board[: max(1, rows)]
    return float(np.count_nonzero(top_rows))


def calculate_lower_fill_score(board: np.ndarray):
    score = 0.0
    for row in range(BOARD_ROWS):
        score += float(np.count_nonzero(board[row])) * (row + 1)
    return score


def calculate_dense_row_reward(board: np.ndarray):
    dense_reward = 0.0
    almost_clear_reward = 0.0

    for row in range(BOARD_ROWS):
        filled = int(np.count_nonzero(board[row]))
        if filled == 0:
            continue

        row_weight = 1.0 + (row / max(1, BOARD_ROWS - 1)) * 1.4
        occupancy = filled / BOARD_COLS
        dense_reward += occupancy * occupancy * row_weight

        if filled >= BOARD_COLS - 3:
            almost_clear_reward += (filled - (BOARD_COLS - 4)) * row_weight

    return dense_reward, almost_clear_reward


def evaluate_board(board: np.ndarray, lines_cleared: int):
    heights = calculate_column_heights(board)
    holes, hole_depth, covered_holes = calculate_holes(board)
    row_transitions, col_transitions = calculate_transitions(board)
    well_penalty = calculate_well_penalty(heights)

    aggregate_height = sum(heights)
    bumpiness = sum(
        abs(heights[idx] - heights[idx + 1]) for idx in range(len(heights) - 1)
    )

    # Classic Dellacherie / El-Osmani inspired weights
    return (
        lines_cleared * 34.18
        - aggregate_height * 1.30
        - holes * 38.99
        - bumpiness * 1.84
        - row_transitions * 3.21
        - col_transitions * 9.34
        - well_penalty * 3.38
    )


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
    return {
        "score": score,
        "lines_cleared": lines_cleared,
        "row": drop_row,
        "board": new_board,
    }


def extract_visible_grid(board_crop, debug=False):
    import cv2

    hsv = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    debug_cells = []

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center_x = GRID_LEFT + (col + 0.5) * CELL_WIDTH
            center_y = GRID_TOP + (row + 0.5) * CELL_HEIGHT
            patch_half_width = CELL_WIDTH * 0.18
            patch_half_height = CELL_HEIGHT * 0.18
            x1 = max(0, int(center_x - patch_half_width))
            x2 = min(board_crop.shape[1], int(center_x + patch_half_width))
            y1 = max(0, int(center_y - patch_half_height))
            y2 = min(board_crop.shape[0], int(center_y + patch_half_height))
            if x2 <= x1 or y2 <= y1:
                continue

            hsv_patch = hsv[y1:y2, x1:x2]
            v_mean = float(np.mean(hsv_patch[:, :, 2]))
            s_mean = float(np.mean(hsv_patch[:, :, 1]))

            # The background grids and shadows possess very low saturation and lower values.
            # Real blocks are bright (V>155) and colorful (S>80).
            if (
                v_mean >= REAL_BLOCK_VALUE_THRESHOLD
                and s_mean >= REAL_BLOCK_SATURATION_THRESHOLD
            ):
                grid[row, col] = True

            if debug:
                debug_cells.append((row, col, v_mean, s_mean, grid[row, col]))

    if debug and debug_cells:
        filled = [(r, c, v, s, f) for r, c, v, s, f in debug_cells if f]
        borderline = [
            (r, c, v, s, f)
            for r, c, v, s, f in debug_cells
            if not f
            and v >= REAL_BLOCK_VALUE_THRESHOLD * 0.85
            and s >= REAL_BLOCK_SATURATION_THRESHOLD * 0.85
        ]
        if filled:
            print(f"[BoardDebug] filled cells ({len(filled)}):")
            for r, c, v, s, f in filled:
                print(f"  ({r},{c}) V={v:.1f} S={s:.1f}")
        if borderline:
            print(f"[BoardDebug] borderline cells ({len(borderline)}):")
            for r, c, v, s, f in borderline:
                print(f"  ({r},{c}) V={v:.1f} S={s:.1f}")

    return grid


def dump_board_state(grid, active_cells=None, piece_state=None, filepath="tetris_debug.txt"):
    """将棋盘状态写入调试文件"""
    import os

    lines = []
    lines.append("=== Tetris Board Debug Dump ===")
    lines.append(f"Grid shape: {grid.shape}")
    lines.append(f"Occupied cells: {int(np.count_nonzero(grid))}")

    if active_cells:
        lines.append(f"Active cells: {active_cells}")
    if piece_state:
        lines.append(f"Piece state: {piece_state}")

    lines.append("")
    lines.append("Board (0=empty, 1=filled, A=active):")
    active_set = set(active_cells) if active_cells else set()

    header = "   " + "".join(f"{c:>2}" for c in range(BOARD_COLS))
    lines.append(header)
    for row in range(BOARD_ROWS):
        row_str = f"{row:>2} "
        for col in range(BOARD_COLS):
            if (row, col) in active_set:
                row_str += " A"
            elif grid[row, col]:
                row_str += " 1"
            else:
                row_str += " ."
        lines.append(row_str)

    # 逐列高度
    heights = calculate_column_heights(grid)
    lines.append("")
    lines.append(f"Column heights: {heights}")
    lines.append(f"Aggregate height: {sum(heights)}")
    holes, hole_depth, covered_holes = calculate_holes(grid)
    lines.append(f"Holes: {holes}, depth: {hole_depth}, covered: {covered_holes}")

    content = "\n".join(lines)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[BoardDebug] Board state dumped to {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"[BoardDebug] Failed to dump board state: {e}")

    return content


def looks_like_game_scene(img, play_state=None):
    import cv2

    if img is None or not isinstance(img, np.ndarray):
        return False

    if play_state is not None and play_state.get("piece_state") is not None:
        return True

    x, y, w, h = BOARD_REGION
    board_crop = img[y : y + h, x : x + w]
    if len(board_crop.shape) == 3 and board_crop.shape[2] == 4:
        board_crop = cv2.cvtColor(board_crop, cv2.COLOR_BGRA2BGR)
    if board_crop is None or board_crop.size == 0:
        return False

    gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)
    line_values = []
    interior_values = []

    for col in BOARD_LINE_COLUMNS:
        if 0 <= col < gray.shape[1]:
            line_values.append(float(np.mean(gray[:, col])))
    for row in BOARD_LINE_ROWS:
        if 0 <= row < gray.shape[0]:
            line_values.append(float(np.mean(gray[row, :])))

    for col in BOARD_INTERIOR_COLUMNS:
        if 0 <= col < gray.shape[1]:
            interior_values.append(float(np.mean(gray[:, col])))
    for row in BOARD_INTERIOR_ROWS:
        if 0 <= row < gray.shape[0]:
            interior_values.append(float(np.mean(gray[row, :])))

    if not line_values or not interior_values:
        return False

    line_mean = float(np.mean(line_values))
    interior_mean = float(np.mean(interior_values))
    queue_pieces = play_state.get("queue_pieces", []) if play_state else []
    occupied_count = 0
    if play_state is not None and play_state.get("grid") is not None:
        occupied_count = int(np.count_nonzero(play_state["grid"]))

    if len(queue_pieces) >= 2 and line_mean >= 28.0 and interior_mean <= 72.0:
        return True

    return (
        occupied_count >= 4
        and line_mean >= interior_mean - 8.0
        and interior_mean <= 82.0
    )


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


def identify_active_piece(grid: np.ndarray, prefer_cells=None):
    from .pieces import match_piece_state

    components = collect_components(grid)
    if not components:
        return None

    candidates = []
    for component in components:
        if len(component) != 4:
            continue

        piece_state = match_piece_state(component)
        if piece_state is None:
            continue

        component_cells = set(component)
        support_count = 0
        for row, col in component:
            next_row = row + 1
            if next_row >= BOARD_ROWS:
                support_count += 1
                continue
            if grid[next_row, col] and (next_row, col) not in component_cells:
                support_count += 1

        candidates.append(
            (
                support_count,
                min(row for row, _ in component),
                -max(row for row, _ in component),
                component,
            )
        )

    if candidates:
        if prefer_cells:
            prefer_set = set(prefer_cells)
            candidates.sort(
                key=lambda item: (
                    -len(set(item[3]).intersection(prefer_set)),
                    item[0],
                    item[1],
                    item[2],
                )
            )
        else:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    prefer_set = set(prefer_cells) if prefer_cells else None
    row_lo = 0
    row_hi = min(6, BOARD_ROWS)
    if prefer_cells:
        min_prefer_row = min(row for row, _ in prefer_cells)
        max_prefer_row = max(row for row, _ in prefer_cells)
        row_lo = max(0, min_prefer_row - 2)
        row_hi = min(BOARD_ROWS, max_prefer_row + 3)

    fallback_candidates = []
    for piece_name, rotations in PIECES.items():
        for shape in rotations:
            shape_height = max(row for row, _ in shape) + 1
            shape_width = max(col for _, col in shape) + 1
            for row in range(row_lo, row_hi - shape_height + 1):
                for col in range(0, BOARD_COLS - shape_width + 1):
                    cells = []
                    matched = True
                    for row_offset, col_offset in shape:
                        cell_row = row + row_offset
                        cell_col = col + col_offset
                        if not grid[cell_row, cell_col]:
                            matched = False
                            break
                        cells.append((cell_row, cell_col))
                    if not matched:
                        continue

                    cell_set = set(cells)
                    support_count = 0
                    for cell_row, cell_col in cells:
                        next_row = cell_row + 1
                        if next_row >= BOARD_ROWS:
                            support_count += 1
                            continue
                        if (
                            grid[next_row, cell_col]
                            and (next_row, cell_col) not in cell_set
                        ):
                            support_count += 1

                    if support_count >= 4:
                        continue

                    overlap = (
                        len(cell_set.intersection(prefer_set))
                        if prefer_set is not None
                        else 0
                    )
                    fallback_candidates.append(
                        (
                            support_count,
                            min(row for row, _ in cells),
                            -max(row for row, _ in cells),
                            overlap,
                            cells,
                        )
                    )

    if not fallback_candidates:
        return None

    if prefer_set:
        fallback_candidates.sort(key=lambda item: (-item[3], item[0], item[1], item[2]))
    else:
        fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return fallback_candidates[0][4]
