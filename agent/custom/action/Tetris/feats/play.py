import time

import numpy as np

from ...Common.utils import get_image
from ..utils.board import (
    BOARD_COLS,
    BOARD_ROWS,
    extract_board_crop,
    extract_visible_grid,
    identify_active_piece,
    simulate_drop,
    evaluate_board,
)
from ..utils.pieces import PIECES, match_piece_state, rotation_distance
from ..utils.scene import SceneGate, EXIT_REGION, PREPARE_ONE_CLICK_POINT, PREPARE_ONE_MULTI_CLICK_POINT, PREPARE_TWO_CLICK_POINT, VK_A, VK_D, VK_F, VK_J, VK_K, VK_ESC, VK_S, VK_SPACE


class TetrisGamePlayer:
    def __init__(self):
        self.scene_gate = SceneGate()
        self.mode = "single"

    def run(self, controller, tasker, mode="single"):
        self.mode = mode
        print(f"=== Auto Tetris Started | mode={self.mode} ===")
        loop_count = 1

        if not self._normalize_scene_for_round(controller, tasker):
            print("Failed to normalize Tetris scene before entering a round.")
            return False

        for round_index in range(loop_count):
            if tasker.stopping:
                return False

            print(f"=== Tetris round {round_index + 1}/{loop_count} ===")
            if not self._ensure_game_session(controller, tasker):
                return False
            if not self._run_single_round(controller, tasker):
                return False
            if not self._sleep_with_stop(tasker, 1.0):
                return False

        print("=== Auto Tetris Finished ===")
        return True

    def _sleep_with_stop(self, tasker, seconds: float) -> bool:
        end_at = time.time() + seconds
        while time.time() < end_at:
            if tasker.stopping:
                return False
            time.sleep(min(0.1, end_at - time.time()))
        return True

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

        grid = extract_visible_grid(board_crop)
        active_cells = identify_active_piece(grid)
        piece_state = match_piece_state(active_cells) if active_cells else None
        queue_pieces = self.scene_gate.read_piece_queue(img)

        return {
            "board_crop": board_crop,
            "grid": grid,
            "active_cells": active_cells,
            "piece_state": piece_state,
            "queue_pieces": queue_pieces,
        }

    def _classify_scene(self, img, play_state=None):
        return self.scene_gate.classify_scene(img, play_state)

    def _wait_for_scene_names(
        self, controller, tasker, expected_names, timeout_seconds=6.0, stable_hits=2,
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

    def _normalize_scene_for_round(self, controller, tasker, timeout_seconds=6.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.2):
                    return False
                continue

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            if scene["name"] in (
                "world_prompt", "prepare_one", "prepare_two",
                "game_active", "game_idle", "exit",
            ):
                return True

            print("Tetris start scene is unknown, pressing ESC to normalize.")
            self._press_escape(controller)
            if not self._sleep_with_stop(tasker, 0.8):
                return False

        return False

    def _recover_from_scene(self, controller, tasker, scene: dict) -> bool:
        scene_name = scene["name"]
        if scene_name == "world_prompt":
            print(f"Recovery: world prompt detected, pressing F. score={scene['score']:.2f}")
            self._tap_key(controller, VK_F)
            return (
                self._wait_for_scene_names(
                    controller, tasker,
                    {"prepare_one", "prepare_two", "game_active", "game_idle"},
                    timeout_seconds=6.0,
                )
                is not None
            )

        if scene_name == "prepare_one":
            click_point = PREPARE_ONE_MULTI_CLICK_POINT if self.mode == "multiple" else PREPARE_ONE_CLICK_POINT
            mode_label = "多人" if self.mode == "multiple" else "单人"
            print(f"Recovery: {mode_label}模式入口 detected, clicking. score={scene['score']:.2f}")
            self._click_point(controller, *click_point)
            return (
                self._wait_for_scene_names(
                    controller, tasker,
                    {"prepare_two", "game_active", "game_idle"},
                    timeout_seconds=6.0,
                )
                is not None
            )

        if scene_name == "prepare_two":
            print(f"Recovery: start-match scene detected, clicking. score={scene['score']:.2f}")
            self._click_point(controller, *PREPARE_TWO_CLICK_POINT)
            return (
                self._wait_for_scene_names(
                    controller, tasker,
                    {"game_active", "game_idle", "exit"},
                    timeout_seconds=12.0,
                )
                is not None
            )

        if scene_name == "exit":
            print(f"Recovery: exit button detected, clicking. score={scene['score']:.2f}")
            self._click_template(controller, scene["x"], scene["y"], scene["template"])
            return self._wait_until_exit_to_world(controller, tasker, timeout_seconds=6.0)

        if scene_name == "game_idle":
            return self._sleep_with_stop(tasker, 0.08)

        print("Recovery: unknown Tetris scene, pressing ESC to go back.")
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
            if scene["name"] == "world_prompt":
                return True

            if scene["name"] == "exit":
                self._click_template(controller, scene["x"], scene["y"], scene["template"])
                if self._wait_until_exit_to_world(controller, tasker, timeout_seconds=4.0):
                    return True
                continue

            self._press_escape(controller)
            if not self._sleep_with_stop(tasker, 0.8):
                return False

        return False

    def _wait_until_exit_to_world(self, controller, tasker, timeout_seconds=8.0) -> bool:
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
            if scene["name"] == "world_prompt":
                return True
            if not self._sleep_with_stop(tasker, 0.2):
                return False
        return False

    def _ensure_game_session(self, controller, tasker, timeout_seconds=20.0) -> bool:
        start_time = time.time()
        unknown_since = None

        while time.time() - start_time < timeout_seconds:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.2):
                    return False
                continue

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]

            if scene_name in ("game_active", "game_idle"):
                return True

            if scene_name == "world_prompt":
                if not self._recover_from_scene(controller, tasker, scene):
                    return False
                continue

            if scene_name == "unknown":
                if unknown_since is None:
                    unknown_since = time.time()
                elif time.time() - unknown_since >= 2.0:
                    print("Tetris scene unknown during session recovery, retrying.")
                    unknown_since = None
                    if not self._back_to_world_from_anywhere(controller, tasker, max_attempts=2):
                        return False
                    continue
                if not self._sleep_with_stop(tasker, 0.2):
                    return False
                continue

            unknown_since = None
            if not self._recover_from_scene(controller, tasker, scene):
                return False

        print("Failed to enter Tetris game scene within timeout.")
        return False

    def _attempt_round_recovery(self, controller, tasker, reason: str) -> bool:
        print(f"{reason} Trying to recover Tetris flow automatically.")
        if not self._back_to_world_from_anywhere(controller, tasker, max_attempts=2):
            return False
        return self._ensure_game_session(controller, tasker, timeout_seconds=10.0)

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
        self, board: np.ndarray, queue_pieces: list[str],
        depth=0, max_depth=4, beam_width=8,
    ):
        if not queue_pieces:
            return None

        piece_name = queue_pieces[0]
        candidates = []
        for rotation_index, shape in enumerate(PIECES[piece_name]):
            width = max(col for _, col in shape) + 1
            for target_col in range(0, BOARD_COLS - width + 1):
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue
                candidates.append({
                    "rotation": rotation_index,
                    "target_col": target_col,
                    "score": result["score"],
                    "board": result["board"],
                    "piece": piece_name,
                })

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        search_candidates = candidates[:beam_width]

        best_choice = None
        for candidate in search_candidates:
            total_score = candidate["score"]
            if depth + 1 < max_depth and len(queue_pieces) > 1:
                future = self._search_best_queue_move(
                    candidate["board"], queue_pieces[1:],
                    depth=depth + 1, max_depth=max_depth, beam_width=beam_width,
                )
                if future is not None:
                    total_score += future["total_score"] * 0.62

            enriched = dict(candidate)
            enriched["total_score"] = total_score
            if best_choice is None or enriched["total_score"] > best_choice["total_score"]:
                best_choice = enriched

        return best_choice

    def _choose_best_current_piece_move(
        self, board: np.ndarray, piece_state: dict, planning_queue: list[str],
    ):
        piece_name = piece_state["piece"]
        future_queue = planning_queue[1:] if planning_queue[:1] == [piece_name] else planning_queue
        best_move = None

        for rotation_index, shape in enumerate(PIECES[piece_name]):
            width = max(col for _, col in shape) + 1
            for target_col in range(0, BOARD_COLS - width + 1):
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue

                future_bonus = 0.0
                if future_queue:
                    future_move = self._search_best_queue_move(
                        result["board"], future_queue,
                        max_depth=min(4, len(future_queue)), beam_width=8,
                    )
                    if future_move is not None:
                        future_bonus = future_move["total_score"] * 0.58

                rot_dist = rotation_distance(piece_name, piece_state["rotation"], rotation_index)
                shift_distance = abs(target_col - piece_state["col"])
                execution_penalty = rot_dist * 0.14 + shift_distance * 0.025
                if piece_name == "I" and rot_dist > 0:
                    execution_penalty += 0.08
                if rot_dist > 0 and shift_distance > 3:
                    execution_penalty += 0.05

                move = {
                    "piece": piece_name,
                    "rotation": rotation_index,
                    "target_col": target_col,
                    "score": result["score"],
                    "total_score": result["score"] + future_bonus - execution_penalty,
                    "lines_cleared": result["lines_cleared"],
                    "future_bonus": future_bonus,
                    "execution_penalty": execution_penalty,
                }
                if best_move is None or move["total_score"] > best_move["total_score"]:
                    best_move = move

        return best_move

    def _apply_move_with_feedback(self, controller, tasker, piece_state, best_move):
        piece_name = piece_state["piece"]
        rotation_count = len(PIECES[piece_name])
        target_rotation = best_move["rotation"]
        target_col = best_move["target_col"]
        deadline = time.time() + 2.6
        hard_drop_sent = False
        post_drop_deadline = None

        while time.time() < deadline:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.04):
                    return False
                continue

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]

            if hard_drop_sent:
                if scene_name in ("exit", "game_idle"):
                    return True
                if play_state is None or play_state["piece_state"] is None:
                    if post_drop_deadline is not None and time.time() >= post_drop_deadline:
                        return True
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                    continue
            else:
                if scene_name in ("world_prompt", "prepare_one", "prepare_two", "exit"):
                    return False
                if scene_name not in ("game_active", "game_idle") or play_state is None:
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                    continue

            current_piece_state = play_state["piece_state"]
            if current_piece_state is None:
                if hard_drop_sent:
                    return True
                if not self._sleep_with_stop(tasker, 0.04):
                    return False
                continue

            if current_piece_state["piece"] != piece_name:
                return True

            current_rotation = current_piece_state["rotation"]
            current_col = current_piece_state["col"]
            if current_rotation == target_rotation and current_col == target_col:
                if not hard_drop_sent:
                    self._tap_key(controller, VK_SPACE, hold=0.06)
                    hard_drop_sent = True
                    post_drop_deadline = time.time() + 0.9
                    if not self._sleep_with_stop(tasker, 0.08):
                        return False
                    continue
                if post_drop_deadline is not None and time.time() >= post_drop_deadline:
                    return True
                if not self._sleep_with_stop(tasker, 0.04):
                    return False
                continue

            clockwise_steps = (target_rotation - current_rotation) % rotation_count
            counterclockwise_steps = (current_rotation - target_rotation) % rotation_count

            if current_rotation != target_rotation:
                correction_key = VK_K if clockwise_steps <= counterclockwise_steps else VK_J
                self._tap_key(controller, correction_key, hold=0.05)
                if not self._sleep_with_stop(tasker, 0.085):
                    return False
                continue

            if current_col != target_col:
                correction_key = VK_D if current_col < target_col else VK_A
                self._tap_key(controller, correction_key, hold=0.05)
                if not self._sleep_with_stop(tasker, 0.08):
                    return False
                continue

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        return False

    def _run_single_round(self, controller, tasker) -> bool:
        if not self._normalize_scene_for_round(controller, tasker):
            print("Failed to normalize Tetris scene before entering a round.")
            return False

        if not self._ensure_game_session(controller, tasker):
            return False

        last_piece_signature = None
        round_start = time.time()
        non_active_since = None

        while time.time() - round_start < 900:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            play_state = self._scan_play_state(img)
            scene = self._classify_scene(img, play_state)
            scene_name = scene["name"]

            if scene_name == "exit":
                print(f"Detected exit button, leaving match. score={scene['score']:.2f}")
                self._click_template(controller, scene["x"], scene["y"], self.scene_gate.exit_tpl)
                self._wait_until_exit_to_world(controller, tasker)
                return True

            if scene_name != "game_active" or play_state is None or play_state["piece_state"] is None:
                if non_active_since is None:
                    non_active_since = time.time()
                elif time.time() - non_active_since >= 2.0:
                    recovered = self._attempt_round_recovery(
                        controller, tasker,
                        f"Tetris scene drifted to {scene_name}.",
                    )
                    if not recovered:
                        return False
                    non_active_since = None
                    last_piece_signature = None
                    if not self._sleep_with_stop(tasker, 0.4):
                        return False
                    continue

                if not self._recover_from_scene(controller, tasker, scene):
                    return False
                continue

            non_active_since = None

            grid = play_state["grid"]
            active_cells = play_state["active_cells"]
            piece_state = play_state["piece_state"]
            queue_pieces = play_state["queue_pieces"]

            if piece_state["cells"] == last_piece_signature:
                if not self._sleep_with_stop(tasker, 0.03):
                    return False
                continue

            settled_board = grid.copy()
            for row, col in active_cells:
                settled_board[row, col] = False

            if queue_pieces:
                print(f"Queue(bottom->top)={queue_pieces}")

            if queue_pieces and queue_pieces[0] == piece_state["piece"]:
                planning_queue = queue_pieces[:3]
            elif queue_pieces:
                try:
                    aligned_index = queue_pieces.index(piece_state["piece"])
                    planning_queue = queue_pieces[aligned_index : aligned_index + 3]
                except ValueError:
                    planning_queue = [piece_state["piece"], *queue_pieces[:2]]
            else:
                planning_queue = [piece_state["piece"]]

            planning_queue = planning_queue[:4]

            best_move = self._choose_best_current_piece_move(
                settled_board, piece_state, planning_queue,
            )
            if best_move is None:
                best_move = self._find_best_move(settled_board, piece_state["piece"])
            if best_move is None:
                print("No valid move found, waiting for next stable frame.")
                if not self._sleep_with_stop(tasker, 0.06):
                    return False
                continue

            print(
                "Piece=%s rot=%s col=%s -> target_rot=%s target_col=%s score=%.2f penalty=%.2f"
                % (
                    piece_state["piece"],
                    piece_state["rotation"],
                    piece_state["col"],
                    best_move["rotation"],
                    best_move["target_col"],
                    best_move.get("total_score", best_move["score"]),
                    best_move.get("execution_penalty", 0.0),
                )
            )
            move_applied = self._apply_move_with_feedback(controller, tasker, piece_state, best_move)
            if move_applied:
                last_piece_signature = piece_state["cells"]
            else:
                print("Move did not reach target position, retrying with a fresh board state.")
            if not self._sleep_with_stop(tasker, 0.04):
                return False

        print("Tetris round timed out.")
        return False
