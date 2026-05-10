import time

import numpy as np

from ...Common.utils import get_image
from ..utils.board import (
    BOARD_COLS,
    BOARD_ROWS,
    extract_board_crop,
    simulate_drop,
    evaluate_board,
)
from ..utils.pieces import PIECES, rotation_distance
from ..utils.scene import (
    SceneGate,
    VK_A,
    VK_D,
    VK_J,
    VK_K,
)
from ..utils.scene_detector import TetrisSceneDetector


class TetrisGamePlayer:
    def __init__(self):
        self.scene_gate = SceneGate()
        self.scene_detector = TetrisSceneDetector()
        self.context = None
        self.mode = "single"
        self.last_active_cells = None
        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0
        self.drop_ready_hits = 0

        self.internal_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self.current_piece_name = None
        self.current_rotation = 0
        self.current_col = 0
        self.current_row = 0
        self.current_cells = None
        self.queue_pieces_state = []

        self.new_piece_roi = (474, 50, 295, 60)

    def reset(self):
        self.last_active_cells = None
        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0
        self.drop_ready_hits = 0
        self.internal_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self.current_piece_name = None
        self.current_rotation = 0
        self.current_col = 0
        self.current_row = 0
        self.current_cells = None
        self.queue_pieces_state = []

    def play_round(self, controller, tasker) -> bool:
        self.reset()
        round_start = time.time()
        last_piece_signature = None
        skip_count = 0

        print(f"=== Tetris Round Started | mode={self.mode} ===")

        while time.time() - round_start < 900:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.02):
                    return False
                continue

            if time.time() - round_start >= 5.0 and self._is_result_screen(img):
                print("Result screen detected, round ended.")
                self._dismiss_result(controller)
                return True

            if self.current_piece_name is None:
                if not self._update_active_piece_state(img):
                    if not self._sleep_with_stop(tasker, 0.02):
                        return False
                    continue
                print(f"[NewPiece] First piece: {self.current_piece_name}")

            current_signature = (
                self.current_piece_name,
                self.current_rotation,
                self.current_col,
            )
            if current_signature == last_piece_signature:
                skip_count += 1
                if skip_count >= 10:
                    print("Same piece signature repeated too long, forcing re-evaluation.")
                    last_piece_signature = None
                    skip_count = 0
                else:
                    if not self._sleep_with_stop(tasker, 0.02):
                        return False
                    continue

            skip_count = 0

            piece_state = {
                "piece": self.current_piece_name,
                "rotation": self.current_rotation,
                "col": self.current_col,
                "row": self.current_row,
                "cells": self.current_cells,
            }
            settled_board = self.internal_board.copy()

            if self.queue_pieces_state:
                planning_queue = [self.current_piece_name, *self.queue_pieces_state[:5]]
            else:
                planning_queue = [self.current_piece_name]
            planning_queue = planning_queue[:6]

            best_move = self._choose_best_current_piece_move(
                settled_board, piece_state, planning_queue,
            )
            if best_move is None:
                best_move = self._find_best_move(settled_board, self.current_piece_name)
            if best_move is None:
                print("No valid move found, waiting.")
                if not self._sleep_with_stop(tasker, 0.04):
                    return False
                continue

            print(
                "Piece=%s rot=%s col=%s -> target_rot=%s target_col=%s score=%.2f"
                % (
                    self.current_piece_name,
                    self.current_rotation,
                    self.current_col,
                    best_move["rotation"],
                    best_move["target_col"],
                    best_move.get("total_score", best_move["score"]),
                )
            )

            planned_result = self._apply_internal_drop(
                settled_board, self.current_piece_name,
                best_move["rotation"], best_move["target_col"], apply=False,
            )

            self._rotate_and_standardize(controller, best_move["rotation"])
            self._apply_move_no_feedback(controller, best_move["rotation"], best_move["target_col"])

            next_piece_info = self._detect_next_piece(controller, tasker)
            if next_piece_info is None:
                return False
            if next_piece_info == "result":
                print("Result screen detected, round ended.")
                return True

            if planned_result is not None:
                self.internal_board = planned_result["board"]
                self._log_internal_board()

            lines_cleared = best_move.get("lines_cleared", 0)
            if planned_result is not None:
                lines_cleared = planned_result.get("lines_cleared", lines_cleared)

            if lines_cleared > 0:
                now = time.time()
                if now - self.last_clear_time < 3.0:
                    self.combo_count += 1
                else:
                    self.combo_count = 1
                self.last_clear_time = now
                self.total_lines_cleared += lines_cleared
                print(f"[Stats] Lines cleared: {lines_cleared}, Total: {self.total_lines_cleared}")
            else:
                if time.time() - self.last_clear_time > 5.0:
                    self.combo_count = 0

            self.current_piece_name = next_piece_info["piece"]
            self.current_rotation = next_piece_info["rotation"]
            self.current_col = next_piece_info["col"]
            self.current_row = next_piece_info["row"]
            self.current_cells = next_piece_info["cells"]
            self.last_active_cells = next_piece_info["cells"]
            last_piece_signature = None

            img2 = self._safe_get_image(controller)
            if img2 is not None:
                self._update_queue_pieces(img2)

            if not self._sleep_with_stop(tasker, 0.02):
                return False

        print("Tetris round timed out.")
        return False

    def _sleep_with_stop(self, tasker, seconds: float) -> bool:
        end_at = time.time() + seconds
        while time.time() < end_at:
            if tasker.stopping:
                return False
            time.sleep(min(0.01, end_at - time.time()))
        return True

    def _detect_next_piece(self, controller, tasker):
        while True:
            if tasker.stopping:
                return None
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.01):
                    return None
                continue

            if self._is_result_screen(img):
                print("Result screen detected while waiting for next piece.")
                self._dismiss_result(controller)
                return "result"

            if not self._is_drop_ready(img):
                if not self._sleep_with_stop(tasker, 0.01):
                    return None
                continue

            match = self.scene_gate.match_active_piece_in_region(
                img, self.new_piece_roi
            )
            if match is not None:
                print(
                    f"[NewPiece] Template matched {match['piece']} score={match['score']:.2f}, new piece spawned"
                )

                base_rotation = 0
                shape = PIECES[match["piece"]][base_rotation]
                piece_info = {
                    "piece": match["piece"],
                    "rotation": base_rotation,
                    "row": 0,
                    "col": 3,
                    "cells": tuple(sorted((r, 3 + c) for r, c in shape)),
                }
                return piece_info

            if not self._sleep_with_stop(tasker, 0.01):
                return None

    def _dismiss_result(self, controller):
        controller.post_key_down(27)
        time.sleep(0.05)
        controller.post_key_up(27)

    def _tap_key(self, controller, key_code: int, hold: float = 0.02):
        controller.post_key_down(key_code)
        time.sleep(hold)
        controller.post_key_up(key_code)

    def _safe_get_image(self, controller):
        try:
            return get_image(controller)
        except Exception as exc:
            print(f"Tetris screencap failed: {exc}")
            return None

    def _scan_play_state(self, img):
        if img is None or not isinstance(img, np.ndarray):
            return None

        from ..utils.board import extract_board_crop

        board_crop = extract_board_crop(img)
        if board_crop is None or board_crop.size == 0:
            return None

        if not self._is_drop_ready(img):
            return {
                "board_crop": board_crop,
                "grid": None,
                "active_cells": None,
                "piece_state": None,
                "queue_pieces": [],
            }

        match = self.scene_gate.match_active_piece_in_region(img, self.new_piece_roi)
        piece_state = None
        active_cells = None

        if match is not None:
            base_rotation = 0
            shape = PIECES[match["piece"]][base_rotation]

            col = 3
            row = 0

            active_cells = [(row + r, col + c) for r, c in shape]
            piece_state = {
                "piece": match["piece"],
                "rotation": base_rotation,
                "row": row,
                "col": col,
                "cells": tuple(sorted(active_cells)),
            }

        queue_pieces = self.scene_gate.read_piece_queue(img)

        if active_cells is not None:
            self.last_active_cells = active_cells

        return {
            "board_crop": board_crop,
            "grid": None,
            "active_cells": active_cells,
            "piece_state": piece_state,
            "queue_pieces": queue_pieces,
        }

    def _is_drop_ready(self, img) -> bool:
        if self.context is not None:
            result = self.scene_detector.check_drop(self.context, img)
            matched = result is not None
        else:
            matched, score, _, _ = self.scene_gate._find_drop_button(img)
        if matched:
            self.drop_ready_hits = min(self.drop_ready_hits + 1, 3)
        else:
            self.drop_ready_hits = 0
        return self.drop_ready_hits >= 2

    def _is_result_screen(self, img) -> bool:
        if self.context is None:
            matched, score, _, _ = self.scene_gate._find_matchend(img)
            return matched
        return self.scene_detector.check_matchend(self.context, img) is not None

    def _log_internal_board(self):
        rows = []
        for r in range(BOARD_ROWS):
            row_cells = [
                "#" if self.internal_board[r, c] else "." for c in range(BOARD_COLS)
            ]
            rows.append("".join(row_cells))
        print("[Board] internal state:\n" + "\n".join(rows))

    def _apply_move_no_feedback(self, controller, target_rotation, target_col):
        rotation_count = len(PIECES[self.current_piece_name])

        clockwise_steps = (target_rotation - self.current_rotation) % rotation_count
        counterclockwise_steps = (
            self.current_rotation - target_rotation
        ) % rotation_count

        rotated = False
        if clockwise_steps <= counterclockwise_steps:
            for _ in range(clockwise_steps):
                self._tap_key(controller, VK_K, hold=0.02)
                time.sleep(0.02)
                self.current_rotation = (self.current_rotation + 1) % rotation_count
                rotated = True
        else:
            for _ in range(counterclockwise_steps):
                self._tap_key(controller, VK_J, hold=0.02)
                time.sleep(0.02)
                self.current_rotation = (self.current_rotation - 1) % rotation_count
                rotated = True

        col_diff = target_col - self.current_col
        if col_diff > 0:
            for _ in range(col_diff):
                self._tap_key(controller, VK_D, hold=0.02)
                time.sleep(0.02)
                self.current_col += 1
        elif col_diff < 0:
            for _ in range(-col_diff):
                self._tap_key(controller, VK_A, hold=0.02)
                time.sleep(0.02)
                self.current_col -= 1

        print(
            f"[ApplyMove] piece={self.current_piece_name} rot={self.current_rotation} col={self.current_col}"
        )

    def _rotate_and_standardize(self, controller, target_rotation):
        rotation_count = len(PIECES[self.current_piece_name])
        actual_rotation = target_rotation % rotation_count

        if actual_rotation != self.current_rotation:
            clockwise_steps = (actual_rotation - self.current_rotation) % rotation_count
            counterclockwise_steps = (self.current_rotation - actual_rotation) % rotation_count
            if clockwise_steps <= counterclockwise_steps:
                for _ in range(clockwise_steps):
                    self._tap_key(controller, VK_K, hold=0.02)
                    time.sleep(0.01)
            else:
                for _ in range(counterclockwise_steps):
                    self._tap_key(controller, VK_J, hold=0.02)
                    time.sleep(0.01)

        for _ in range(10):
            self._tap_key(controller, VK_A, hold=0.02)
            time.sleep(0.01)
        self.current_col = 0

        print(
            f"[RotateAndStd] piece={self.current_piece_name} rot={self.current_rotation} col={self.current_col}"
        )

    def _update_active_piece_state(self, img) -> bool:
        play_state = self._scan_play_state(img)
        if play_state is None:
            return False

        piece_state = play_state.get("piece_state")
        if piece_state is None:
            return False

        self.current_piece_name = piece_state["piece"]
        self.current_rotation = piece_state["rotation"]
        self.current_col = piece_state["col"]
        self.current_row = piece_state["row"]
        self.current_cells = piece_state["cells"]

        self.queue_pieces_state = play_state.get("queue_pieces", [])
        if self.queue_pieces_state:
            print(f"Queue(bottom->top)={self.queue_pieces_state}")

        active_cells = play_state.get("active_cells")
        if active_cells is not None:
            self.last_active_cells = active_cells

        return True

    def _update_queue_pieces(self, img):
        play_state = self._scan_play_state(img)
        if play_state is None:
            return
        queue = play_state.get("queue_pieces", [])
        if queue:
            self.queue_pieces_state = queue
            print(f"Queue(bottom->top)={self.queue_pieces_state}")

    def _apply_internal_drop(
        self,
        board: np.ndarray,
        piece_name: str,
        rotation: int,
        target_col: int,
        apply: bool = True,
    ):
        shape = PIECES[piece_name][rotation]
        result = simulate_drop(board, shape, target_col)
        if result is None:
            print("[Board] Internal drop failed; keeping previous board state.")
            return None
        if apply:
            self.internal_board = result["board"]
            self._log_internal_board()
        return result

    def _can_place(self, board: np.ndarray, shape, row: int, col: int) -> bool:
        for row_offset, col_offset in shape:
            r = row + row_offset
            c = col + col_offset
            if r < 0 or r >= BOARD_ROWS or c < 0 or c >= BOARD_COLS:
                return False
            if board[r, c]:
                return False
        return True

    def _is_move_feasible(
        self,
        board: np.ndarray,
        piece_name: str,
        from_rotation: int,
        from_col: int,
        from_row: int,
        target_rotation: int,
        target_col: int,
    ) -> bool:
        rotation_count = len(PIECES[piece_name])
        clockwise_steps = (target_rotation - from_rotation) % rotation_count
        counterclockwise_steps = (from_rotation - target_rotation) % rotation_count

        if clockwise_steps <= counterclockwise_steps:
            step = 1
            steps = clockwise_steps
        else:
            step = -1
            steps = counterclockwise_steps

        cur_rotation = from_rotation
        for _ in range(steps):
            cur_rotation = (cur_rotation + step) % rotation_count
            if not self._can_place(
                board, PIECES[piece_name][cur_rotation], from_row, from_col
            ):
                return False

        cur_col = from_col
        delta = target_col - from_col
        move_step = 1 if delta > 0 else -1 if delta < 0 else 0
        for _ in range(abs(delta)):
            cur_col += move_step
            if not self._can_place(
                board, PIECES[piece_name][target_rotation], from_row, cur_col
            ):
                return False

        return True

    def _find_best_move(self, board: np.ndarray, piece_name: str):
        best_move = None
        for rotation_index, shape in enumerate(PIECES[piece_name]):
            width = max(col for _, col in shape) + 1
            for target_col in range(0, BOARD_COLS - width + 1):
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue
                move = {
                    "rotation": rotation_index,
                    "target_col": target_col,
                    "score": result["score"],
                    "lines_cleared": result["lines_cleared"],
                }
                if best_move is None or move["score"] > best_move["score"]:
                    best_move = move
        return best_move

    def _search_best_queue_move(
        self,
        board: np.ndarray,
        queue_pieces: list[str],
        depth=0,
        max_depth=2,
        beam_width=5,
        combo_count=0,
    ):
        if not queue_pieces:
            return None

        occupancy = np.count_nonzero(board) / (BOARD_ROWS * BOARD_COLS)

        adaptive_depth = max_depth
        if occupancy < 0.25:
            adaptive_depth = max_depth
        elif occupancy > 0.55:
            adaptive_depth = min(max_depth + 2, 5)
        elif occupancy > 0.4:
            adaptive_depth = min(max_depth + 1, 4)

        adaptive_beam = beam_width
        if len(queue_pieces) >= 3:
            adaptive_beam = min(beam_width + 2, 8)
        elif occupancy > 0.45:
            adaptive_beam = min(beam_width + 1, 7)

        piece_name = queue_pieces[0]
        candidates = []
        for rotation_index, shape in enumerate(PIECES[piece_name]):
            width = max(col for _, col in shape) + 1
            for target_col in range(0, BOARD_COLS - width + 1):
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue

                is_t_spin = False
                if depth == 0 and piece_name == "T":
                    from ..utils.board import detect_t_spin

                    t_spin_result = detect_t_spin(
                        board,
                        piece_name,
                        rotation_index,
                        target_col,
                        result["row"],
                        was_rotation_move=False,
                    )
                    is_t_spin = t_spin_result["is_t_spin"]

                next_combo = combo_count + 1 if result["lines_cleared"] > 0 else 0
                if depth == 0:
                    eval_score = evaluate_board(
                        result["board"],
                        result["lines_cleared"],
                        dynamic_weights=True,
                        combo_count=next_combo,
                        is_t_spin=is_t_spin,
                    )
                else:
                    from ..utils.board import evaluate_board_fast

                    eval_score = evaluate_board_fast(
                        result["board"], result["lines_cleared"], combo_count=next_combo
                    )

                candidates.append(
                    {
                        "rotation": rotation_index,
                        "target_col": target_col,
                        "score": eval_score,
                        "board": result["board"],
                        "piece": piece_name,
                        "lines_cleared": result["lines_cleared"],
                        "is_t_spin": is_t_spin,
                    }
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        search_candidates = candidates[:adaptive_beam]

        best_choice = None
        for candidate in search_candidates:
            total_score = candidate["score"]
            next_combo = combo_count + 1 if candidate["lines_cleared"] > 0 else 0
            if depth + 1 < adaptive_depth and len(queue_pieces) > 1:
                future = self._search_best_queue_move(
                    candidate["board"],
                    queue_pieces[1:],
                    depth=depth + 1,
                    max_depth=adaptive_depth,
                    beam_width=adaptive_beam,
                    combo_count=next_combo,
                )
                if future is not None:
                    future_value = future["total_score"]
                    depth_discount = 0.85 ** (depth + 1)
                    future_weight = 0.7 if next_combo > 0 else 0.5
                    if candidate.get("is_t_spin"):
                        future_weight = 0.8
                    total_score = (
                        candidate["score"]
                        + future_value * future_weight * depth_discount
                    )

            enriched = dict(candidate)
            enriched["total_score"] = total_score
            if (
                best_choice is None
                or enriched["total_score"] > best_choice["total_score"]
            ):
                best_choice = enriched

        return best_choice

    def _choose_best_current_piece_move(
        self,
        board: np.ndarray,
        piece_state: dict,
        planning_queue: list[str],
    ):
        piece_name = piece_state["piece"]
        future_queue = planning_queue[1:]

        occupancy = np.count_nonzero(board) / (BOARD_ROWS * BOARD_COLS)

        adaptive_depth = 2
        if occupancy < 0.25:
            adaptive_depth = 2
        elif occupancy > 0.55:
            adaptive_depth = 4
        elif occupancy > 0.4:
            adaptive_depth = 3

        beam_width = 6 if len(future_queue) >= 2 else 5

        best_move = None

        for rotation_index, shape in enumerate(PIECES[piece_name]):
            width = max(col for _, col in shape) + 1
            for target_col in range(0, BOARD_COLS - width + 1):
                if not self._is_move_feasible(
                    board,
                    piece_name,
                    piece_state["rotation"],
                    piece_state["col"],
                    piece_state["row"],
                    rotation_index,
                    target_col,
                ):
                    continue
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue

                is_t_spin = False
                from ..utils.board import detect_t_spin

                if piece_name == "T":
                    rot_dist = rotation_distance(
                        piece_name, piece_state["rotation"], rotation_index
                    )
                    t_spin_result = detect_t_spin(
                        board,
                        piece_name,
                        rotation_index,
                        target_col,
                        result["row"],
                        was_rotation_move=rot_dist > 0,
                    )
                    is_t_spin = t_spin_result["is_t_spin"]

                future_bonus = 0.0
                if future_queue:
                    next_combo = (
                        self.combo_count + 1 if result["lines_cleared"] > 0 else 0
                    )
                    future_move = self._search_best_queue_move(
                        result["board"],
                        future_queue,
                        max_depth=adaptive_depth,
                        beam_width=beam_width,
                        combo_count=next_combo,
                    )
                    if future_move is not None:
                        future_weight = 0.7 if next_combo > 0 else 0.5
                        if is_t_spin:
                            future_weight = 0.8
                        future_bonus = future_move["total_score"] * future_weight

                current_score = evaluate_board(
                    result["board"],
                    result["lines_cleared"],
                    dynamic_weights=True,
                    combo_count=(
                        self.combo_count + 1 if result["lines_cleared"] > 0 else 0
                    ),
                    is_t_spin=is_t_spin,
                )

                rot_dist = rotation_distance(
                    piece_name, piece_state["rotation"], rotation_index
                )
                shift_distance = abs(target_col - piece_state["col"])
                execution_penalty = rot_dist * 0.15 + shift_distance * 0.06

                if rot_dist > 0 and shift_distance > 4:
                    execution_penalty += 0.15

                move = {
                    "piece": piece_name,
                    "rotation": rotation_index,
                    "target_col": target_col,
                    "score": current_score,
                    "total_score": current_score + future_bonus - execution_penalty,
                    "lines_cleared": result["lines_cleared"],
                    "future_bonus": future_bonus,
                    "execution_penalty": execution_penalty,
                    "is_t_spin": is_t_spin,
                }
                if best_move is None or move["total_score"] > best_move["total_score"]:
                    best_move = move

        return best_move
