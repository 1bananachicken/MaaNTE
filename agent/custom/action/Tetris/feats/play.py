import time

import numpy as np

from ...Common.utils import get_image
from ..utils.board import (
    BOARD_COLS,
    BOARD_REGION,
    BOARD_ROWS,
    CELL_HEIGHT,
    CELL_WIDTH,
    extract_board_crop,
    simulate_drop,
    evaluate_board,
    GRID_LEFT,
    GRID_TOP,
)
from ..utils.pieces import PIECES, rotation_distance
from ..utils.scene import (
    SceneGate,
    EXIT_REGION,
    LOADING_REGION,
    MATCHEND_REGION,
    RETURN_REGION,
    PREPARE_ONE_CLICK_POINT,
    PREPARE_ONE_MULTI_CLICK_POINT,
    PREPARE_TWO_CLICK_POINT,
    VK_A,
    VK_D,
    VK_F,
    VK_J,
    VK_K,
    VK_ESC,
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

    def run(self, controller, tasker, mode="single", context=None):
        self.mode = mode
        self.context = context
        self.reset()
        print(f"=== Auto Tetris Started | mode={self.mode} ===")

        initial_scene = self._detect_initial_scene(controller, tasker)
        if initial_scene is None:
            print("Cannot detect Tetris initial scene, ending task.")
            return False

        scene_name = initial_scene["name"]
        print(f"Initial scene detected: {scene_name}")

        if scene_name == "world_no_prompt":
            print(
                "World scene detected but no Tetris entrance prompt found, ending task."
            )
            return False

        if scene_name == "unknown":
            print("Unknown initial scene, ending task.")
            return False

        if scene_name in ("game_active", "game_idle"):
            return self._play_from_game(controller, tasker)

        return self._navigate_to_game_and_play(controller, tasker)

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
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            if time.time() - round_start >= 5.0 and self._is_result_screen(img):
                print("Result screen detected, round ended.")
                return True

            if self.current_piece_name is None:
                if not self._update_active_piece_state(img):
                    if not self._sleep_with_stop(tasker, 0.04):
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
                    if not self._sleep_with_stop(tasker, 0.03):
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
                if not self._sleep_with_stop(tasker, 0.06):
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

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        print("Tetris round timed out.")
        return False

    def _sleep_with_stop(self, tasker, seconds: float) -> bool:
        end_at = time.time() + seconds
        while time.time() < end_at:
            if tasker.stopping:
                return False
            time.sleep(min(0.1, end_at - time.time()))
        return True

    def _detect_next_piece(self, controller, tasker):
        while True:
            if tasker.stopping:
                return None
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return None
                continue

            if self._is_result_screen(img):
                print("Result screen detected while waiting for next piece.")
                return "result"

            if not self._is_drop_ready(img):
                if not self._sleep_with_stop(tasker, 0.05):
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

            if not self._sleep_with_stop(tasker, 0.05):
                return None

    def _tap_key(self, controller, key_code: int, hold: float = 0.03):
        controller.post_key_down(key_code)
        time.sleep(hold)
        controller.post_key_up(key_code)

    def _press_escape(self, controller):
        self._tap_key(controller, VK_ESC, hold=0.05)

    def _click_point(self, controller, x: int, y: int):
        try:
            controller.post_click(x, y).wait()
            time.sleep(0.12)
        except Exception as exc:
            print(f"Tetris post_click failed, fallback to touch: {exc}")
            controller.post_touch_down(x, y).wait()
            time.sleep(0.05)
            controller.post_touch_up().wait()
            time.sleep(0.12)

    def _click_template(self, controller, x: int, y: int, template):
        click_x = int(x + template.shape[1] / 2)
        click_y = int(y + template.shape[0] / 2)
        self._click_point(controller, click_x, click_y)

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

    def _wait_for_loading_end(self, controller, tasker, timeout_seconds=30.0) -> str:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if tasker.stopping:
                return "stopped"
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return "stopped"
                continue

            if self.context is not None:
                world_matched = self.scene_detector.check_world_prompt(self.context, img) is not None
                drop_matched = self.scene_detector.check_drop(self.context, img) is not None
                game_matched = self.scene_detector.check_game_marker(self.context, img) is not None
                loading_matched = self.scene_detector.check_loading(self.context, img) is not None
            else:
                world_matched, _, _, _ = self.scene_gate._find_world_prompt(img)
                drop_matched, _, _, _ = self.scene_gate._find_drop_button(img)
                game_matched, _, _, _ = self.scene_gate._find_game_scene_marker(img)
                loading_matched, _, _, _ = self.scene_gate._find_loading(img)

            if world_matched:
                return "world"
            if drop_matched or game_matched:
                return "play"
            if loading_matched:
                if not self._sleep_with_stop(tasker, 0.05):
                    return "stopped"
                continue
            if not self._sleep_with_stop(tasker, 0.05):
                return "stopped"
        print("Loading wait timed out.")
        return "timeout"

    def _wait_for_matching_end(self, controller, tasker, timeout_seconds=90.0) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if tasker.stopping:
                return False
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 1.0):
                    return False
                continue
            if self.context is not None:
                matched = self.scene_detector.check_matching(self.context, img) is not None
            else:
                matched, _, _, _ = self.scene_gate._find_matching(img)
            if not matched:
                return True
            if not self._sleep_with_stop(tasker, 1.0):
                return False
        print("Matching wait timed out.")
        return False

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
                self._tap_key(controller, VK_K, hold=0.03)
                time.sleep(0.03)
                self.current_rotation = (self.current_rotation + 1) % rotation_count
                rotated = True
        else:
            for _ in range(counterclockwise_steps):
                self._tap_key(controller, VK_J, hold=0.03)
                time.sleep(0.03)
                self.current_rotation = (self.current_rotation - 1) % rotation_count
                rotated = True

        col_diff = target_col - self.current_col
        if col_diff > 0:
            for _ in range(col_diff):
                self._tap_key(controller, VK_D, hold=0.03)
                time.sleep(0.03)
                self.current_col += 1
        elif col_diff < 0:
            for _ in range(-col_diff):
                self._tap_key(controller, VK_A, hold=0.03)
                time.sleep(0.03)
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
                    self._tap_key(controller, VK_K, hold=0.03)
                    time.sleep(0.01)
            else:
                for _ in range(counterclockwise_steps):
                    self._tap_key(controller, VK_J, hold=0.03)
                    time.sleep(0.01)
            self.current_rotation = actual_rotation

        for _ in range(10):
            self._tap_key(controller, VK_A, hold=0.03)
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

    def _classify_scene(self, img, play_state=None):
        if self.context is not None:
            return self.scene_detector.classify(self.context, None, play_state, img)
        return self.scene_gate.classify_scene(img, play_state)

    def _wait_for_scene_names(
        self,
        controller,
        tasker,
        expected_names,
        timeout_seconds=6.0,
        stable_hits=2,
    ):
        deadline = time.time() + timeout_seconds
        stable_count = 0
        last_scene_name = ""
        last_scene = None

        while time.time() < deadline:
            if tasker.stopping:
                return None

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.15):
                    return None
                continue

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]
            if scene_name in expected_names:
                if scene_name == last_scene_name:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_scene_name = scene_name
                last_scene = scene
                if stable_count >= stable_hits:
                    return last_scene
            else:
                stable_count = 0
                last_scene_name = scene_name

            if not self._sleep_with_stop(tasker, 0.15):
                return None

        return last_scene

    def _detect_initial_scene(self, controller, tasker, timeout_seconds=6.0):
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if tasker.stopping:
                return None

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.2):
                    return None
                continue

            if self._is_result_screen(img):
                return {"name": "result"}

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]

            if scene_name in (
                "world_prompt",
                "world_no_prompt",
                "prepare_one",
                "prepare_two",
                "game_active",
                "game_idle",
                "exit",
                "loading",
            ):
                return scene

            if not self._sleep_with_stop(tasker, 0.15):
                return None

        return None

    def _navigate_to_game_and_play(self, controller, tasker):
        img = self._safe_get_image(controller)
        if img is None:
            return False

        play_state = self._scan_play_state(img)
        scene = self._classify_scene(img, play_state)
        scene_name = scene["name"]

        if scene_name == "loading":
            print("Waiting for loading screen to finish...")
            result = self._wait_for_loading_end(controller, tasker)
            if result == "play":
                print("Play detected during loading, starting game immediately...")
                return self._play_from_game(controller, tasker, skip_scene_init=True)
            elif result == "world":
                print("World detected during loading, continuing navigation...")
            elif result == "stopped":
                return False
            img = self._safe_get_image(controller)
            if img is None:
                return False
            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]
            if scene_name in ("game_active", "game_idle"):
                return self._play_from_game(controller, tasker)
            if scene_name in ("world_prompt", "world_no_prompt"):
                pass

        if scene_name == "exit":
            print(f"Exit scene at start, clicking exit.")
            self._click_point(controller, scene["x"], scene["y"])
            if not self._wait_until_exit_to_world(controller, tasker):
                return False
            img = self._safe_get_image(controller)
            if img is None:
                return False
            scene = self._classify_scene(img)
            scene_name = scene["name"]

        if scene_name == "world_prompt":
            print(f"World with prompt, pressing F.")
            self._tap_key(controller, VK_F)
            result = self._wait_for_scene_names(
                controller,
                tasker,
                {"prepare_one", "prepare_two"},
                timeout_seconds=6.0,
            )
            if result is None:
                return False
            scene_name = result["name"]
            scene = result

        if scene_name == "prepare_two":
            print("Found prepare_two at start, clicking return to re-select mode.")
            img = self._safe_get_image(controller)
            if img is not None:
                if self.context is not None:
                    ret = self.scene_detector.check_return(self.context, img)
                    if ret:
                        self._click_point(controller, ret["cx"], ret["cy"])
                else:
                    found, score, rx, ry = self.scene_gate._find_return_button(img)
                    if found:
                        click_x = int(rx + self.scene_gate.return_tpl.shape[1] / 2)
                        click_y = int(ry + self.scene_gate.return_tpl.shape[0] / 2)
                        self._click_point(controller, click_x, click_y)
                    result = self._wait_for_scene_names(
                        controller,
                        tasker,
                        {"prepare_one", "world_prompt"},
                        timeout_seconds=6.0,
                    )
                    if result is not None:
                        scene_name = result["name"]
                        scene = result

        if scene_name == "prepare_one":
            click_point = (
                PREPARE_ONE_MULTI_CLICK_POINT
                if self.mode == "multiple"
                else PREPARE_ONE_CLICK_POINT
            )
            mode_label = "多人" if self.mode == "multiple" else "单人"
            print(f"{mode_label}模式入口, clicking.")
            self._click_point(controller, *click_point)
            result = self._wait_for_scene_names(
                controller,
                tasker,
                {"prepare_two", "game_active", "game_idle", "loading"},
                timeout_seconds=6.0,
            )
            if result is None:
                return False
            scene_name = result["name"]
            scene = result

        if scene_name == "prepare_two":
            print(f"Start-match scene, clicking.")
            self._click_point(controller, *PREPARE_TWO_CLICK_POINT)

            if self.mode == "multiple":
                print("Multiple mode: waiting for matching to complete...")
                if not self._sleep_with_stop(tasker, 1.0):
                    return False
                if not self._wait_for_matching_end(controller, tasker):
                    return False

            expected_after_start = {"game_active", "game_idle", "exit", "loading"}
            result = self._wait_for_scene_names(
                controller,
                tasker,
                expected_after_start,
                timeout_seconds=60.0,
            )
            if result is None:
                return False
            scene_name = result["name"]

            if scene_name == "loading":
                print("Loading after match, waiting...")
                result = self._wait_for_loading_end(controller, tasker)
                if result == "play":
                    print("Play detected during loading, starting game immediately...")
                    return self._play_from_game(controller, tasker, skip_scene_init=True)
                elif result == "world":
                    print("World detected during loading after match, continuing navigation...")
                elif result == "stopped":
                    return False
                img = self._safe_get_image(controller)
                if img is None:
                    return False
                play_state = self._scan_play_state(img)
                scene = self._classify_scene(img, play_state)
                scene_name = scene["name"]

        if scene_name in ("game_active", "game_idle"):
            return self._play_from_game(controller, tasker)

        print(f"Unexpected scene after navigation: {scene_name}")
        return False

    _SCENE_LOCK_SEC = 3.0

    def _play_from_game(self, controller, tasker, skip_scene_init=False):
        last_piece_signature = None
        skip_count = 0
        round_start = time.time()
        non_active_since = None
        scene_lock_until = time.time() + self._SCENE_LOCK_SEC
        scene_locked = True
        last_result_check = time.time()

        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0
        self.current_piece_name = None

        if skip_scene_init:
            print("[Play] Skipping scene init, scanning for piece immediately...")

        while time.time() - round_start < 900:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            now = time.time()
            play_state_for_scene = self._scan_play_state(img)

            if now - last_result_check >= 5.0:
                last_result_check = now
                if self._is_result_screen(img):
                    print("Result screen detected, match ended.")
                    return True

            if scene_locked:
                if now < scene_lock_until:
                    if self.current_piece_name is not None:
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                    elif (
                        play_state_for_scene is not None
                        and play_state_for_scene["piece_state"] is not None
                    ):
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                    elif skip_scene_init:
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                    else:
                        scene_locked = False
                elif skip_scene_init:
                    scene_lock_until = now + self._SCENE_LOCK_SEC
                else:
                    scene_locked = False

            if not scene_locked:
                scene = self._classify_scene(img, play_state_for_scene)
                scene_name = scene["name"]

                if scene_name == "exit":
                    print("Exit detected, clicking to leave match.")
                    self._click_point(controller, scene["x"], scene["y"])
                    self._wait_until_exit_to_world(controller, tasker)
                    print("=== Auto Tetris Finished ===")
                    return True

                if scene_name == "game_idle":
                    self.last_active_cells = None
                    if (
                        play_state_for_scene is not None
                        and play_state_for_scene["piece_state"] is not None
                    ):
                        scene_locked = True
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                    else:
                        if not self._sleep_with_stop(tasker, 0.15):
                            return False
                        continue

                elif scene_name == "loading":
                    self.last_active_cells = None
                    print("Loading detected during play, waiting...")
                    result = self._wait_for_loading_end(controller, tasker)
                    if result == "play":
                        print("Play detected during loading, continuing game...")
                        scene_locked = True
                        scene_lock_until = time.time() + self._SCENE_LOCK_SEC
                        continue
                    elif result == "world":
                        print("World detected during play loading, round ended.")
                        return False
                    elif result == "stopped":
                        return False
                    scene_locked = True
                    scene_lock_until = time.time() + self._SCENE_LOCK_SEC
                    continue

                elif scene_name == "world_no_prompt":
                    self.last_active_cells = None
                    print("Drifted to world without prompt during play, ending task.")
                    return False

                elif scene_name in (
                    "game_active",
                    "world_prompt",
                    "prepare_one",
                    "prepare_two",
                    "unknown",
                ):
                    if (
                        scene_name == "game_active"
                        and play_state_for_scene is not None
                        and play_state_for_scene["piece_state"] is not None
                    ):
                        self.last_active_cells = None
                        scene_locked = True
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                    else:
                        self.last_active_cells = None
                        if non_active_since is None:
                            non_active_since = now
                        elif now - non_active_since >= 5.0:
                            recovered = self._attempt_round_recovery(
                                controller,
                                tasker,
                                f"Scene drifted to {scene_name}.",
                            )
                            if not recovered:
                                return False
                            non_active_since = None
                            last_piece_signature = None
                            scene_locked = False
                            if not self._sleep_with_stop(tasker, 0.4):
                                return False
                            continue

                        if not self._recover_from_scene(controller, tasker, scene):
                            return False
                        continue

            if self.current_piece_name is None:
                if not self._update_active_piece_state(img):
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                    continue

                print(
                    f"[NewPiece] First or recovered piece: {self.current_piece_name}"
                )

                scene_locked = True
                scene_lock_until = time.time() + self._SCENE_LOCK_SEC

            non_active_since = None

            current_signature = (
                self.current_piece_name,
                self.current_rotation,
                self.current_col,
            )
            if current_signature == last_piece_signature:
                skip_count += 1
                if skip_count >= 10:
                    print(
                        "Same piece signature repeated too long, forcing re-evaluation."
                    )
                    last_piece_signature = None
                    skip_count = 0
                else:
                    if not self._sleep_with_stop(tasker, 0.03):
                        return False
                    continue

            skip_count = 0
            scene_lock_until = time.time() + self._SCENE_LOCK_SEC

            decision_start = time.time()

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
                settled_board,
                piece_state,
                planning_queue,
            )
            if best_move is None:
                best_move = self._find_best_move(settled_board, self.current_piece_name)
            if best_move is None:
                print("No valid move found, waiting for next stable frame.")
                if not self._sleep_with_stop(tasker, 0.06):
                    return False
                continue

            decision_time = time.time() - decision_start
            if decision_time > 0.3:
                print(f"[Perf] Decision took {decision_time:.3f}s")

            print(
                "Piece=%s rot=%s col=%s -> target_rot=%s target_col=%s score=%.2f penalty=%.2f"
                % (
                    self.current_piece_name,
                    self.current_rotation,
                    self.current_col,
                    best_move["rotation"],
                    best_move["target_col"],
                    best_move.get("total_score", best_move["score"]),
                    best_move.get("execution_penalty", 0.0),
                )
            )

            planned_result = self._apply_internal_drop(
                settled_board,
                self.current_piece_name,
                best_move["rotation"],
                best_move["target_col"],
                apply=False,
            )

            self._rotate_and_standardize(controller, best_move["rotation"])

            self._apply_move_no_feedback(
                controller, best_move["rotation"], best_move["target_col"]
            )
            next_piece_info = self._detect_next_piece(controller, tasker)
            if next_piece_info is None:
                return False
            if next_piece_info == "result":
                print("Result screen detected, match ended.")
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

                if best_move.get("is_t_spin"):
                    print(f"[Special] T-SPIN detected! Combo={self.combo_count}")
                if self.combo_count > 1:
                    print(f"[Special] COMBO x{self.combo_count}!")
                print(f"[Stats] Total lines: {self.total_lines_cleared}")
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

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        print("Tetris round timed out.")
        return False

    def _recover_from_scene(self, controller, tasker, scene: dict) -> bool:
        scene_name = scene["name"]
        if scene_name == "world_prompt":
            print("Recovery: world prompt, pressing F.")
            self._tap_key(controller, VK_F)
            return (
                self._wait_for_scene_names(
                    controller,
                    tasker,
                    {"prepare_one", "prepare_two", "game_active", "game_idle"},
                    timeout_seconds=6.0,
                )
                is not None
            )

        if scene_name == "prepare_one":
            click_point = (
                PREPARE_ONE_MULTI_CLICK_POINT
                if self.mode == "multiple"
                else PREPARE_ONE_CLICK_POINT
            )
            mode_label = "多人" if self.mode == "multiple" else "单人"
            print(f"Recovery: {mode_label}模式入口, clicking.")
            self._click_point(controller, *click_point)
            return (
                self._wait_for_scene_names(
                    controller,
                    tasker,
                    {"prepare_two", "game_active", "game_idle"},
                    timeout_seconds=6.0,
                )
                is not None
            )

        if scene_name == "prepare_two":
            print("Recovery: start-match, clicking.")
            self._click_point(controller, *PREPARE_TWO_CLICK_POINT)
            return (
                self._wait_for_scene_names(
                    controller,
                    tasker,
                    {"game_active", "game_idle", "exit", "loading"},
                    timeout_seconds=12.0,
                )
                is not None
            )

        if scene_name == "exit":
            print("Recovery: exit button, clicking.")
            self._click_point(controller, scene["x"], scene["y"])
            return self._wait_until_exit_to_world(
                controller, tasker, timeout_seconds=6.0
            )

        if scene_name == "loading":
            print("Recovery: loading screen, waiting...")
            result = self._wait_for_loading_end(controller, tasker)
            if result == "play":
                print("Play detected during loading, recovery complete.")
                return True
            elif result == "world":
                print("World detected during loading, recovery failed.")
                return False
            elif result == "stopped":
                return False
            img = self._safe_get_image(controller)
            if img is None:
                return False
            scene = self._classify_scene(img)
            return scene["name"] in ("game_active", "game_idle")

        if scene_name == "game_idle":
            return self._sleep_with_stop(tasker, 0.08)

        img = self._safe_get_image(controller)
        if img is not None:
            if self.context is not None:
                ret = self.scene_detector.check_return(self.context, img)
                if ret:
                    print(f"Recovery: return button found, clicking.")
                    self._click_point(controller, ret["cx"], ret["cy"])
                    return (
                        self._wait_for_scene_names(
                            controller,
                            tasker,
                            {"prepare_one", "world_prompt"},
                            timeout_seconds=4.0,
                        )
                        is not None
                    )
            else:
                found, score, rx, ry = self.scene_gate._find_return_button(img)
                if found:
                    print(
                        f"Recovery: return button found at ({rx},{ry}), clicking. score={score:.2f}"
                    )
                    click_x = int(rx + self.scene_gate.return_tpl.shape[1] / 2)
                    click_y = int(ry + self.scene_gate.return_tpl.shape[0] / 2)
                    self._click_point(controller, click_x, click_y)
                    return (
                        self._wait_for_scene_names(
                            controller,
                            tasker,
                            {"prepare_one", "world_prompt"},
                            timeout_seconds=4.0,
                        )
                        is not None
                    )

        print("Recovery: unknown scene, pressing ESC.")
        self._press_escape(controller)
        return self._sleep_with_stop(tasker, 0.8)

    def _back_to_world_from_anywhere(self, controller, tasker, max_attempts=3) -> bool:
        for _ in range(max_attempts):
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.2):
                    return False
                continue

            scene = self._classify_scene(img)
            if scene["name"] in ("world_prompt", "world_no_prompt"):
                return True

            if scene["name"] == "exit":
                self._click_point(controller, scene["x"], scene["y"])
                if self._wait_until_exit_to_world(
                    controller, tasker, timeout_seconds=4.0
                ):
                    return True
                continue

            if scene["name"] == "loading":
                print("Loading detected while returning to world, waiting...")
                result = self._wait_for_loading_end(controller, tasker)
                if result == "play":
                    continue
                if result == "world":
                    print("World detected during loading, waiting for scene stabilization...")
                    if not self._sleep_with_stop(tasker, 3.0):
                        return False
                    return True
                if result == "timeout":
                    continue
                return False

            self._press_escape(controller)
            if not self._sleep_with_stop(tasker, 0.8):
                return False

        return False

    def _wait_until_exit_to_world(
        self, controller, tasker, timeout_seconds=8.0
    ) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if tasker.stopping:
                return False
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.2):
                    return False
                continue
            scene = self._classify_scene(img)
            if scene["name"] in ("world_prompt", "world_no_prompt"):
                return True
            if not self._sleep_with_stop(tasker, 0.2):
                return False
        return False

    def _attempt_round_recovery(self, controller, tasker, reason: str) -> bool:
        print(f"{reason} Trying to recover Tetris flow automatically.")
        if not self._back_to_world_from_anywhere(controller, tasker, max_attempts=2):
            return False
        img = self._safe_get_image(controller)
        if img is None:
            return False
        scene = self._classify_scene(img)
        if scene["name"] == "world_prompt":
            return self._navigate_to_game_and_play(controller, tasker)
        return False

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
