import time

import numpy as np

from ...Common.utils import get_image
from ..utils.board import (
    BOARD_COLS,
    BOARD_REGION,
    BOARD_ROWS,
    DEBUG_BOARD,
    REAL_BLOCK_VALUE_THRESHOLD,
    extract_board_crop,
    extract_visible_grid,
    identify_active_piece,
    simulate_drop,
    evaluate_board,
)
from ..utils.pieces import PIECES, match_piece_state, rotation_distance
from ..utils.scene import (
    SceneGate,
    EXIT_REGION,
    LOADING_REGION,
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


class TetrisGamePlayer:
    def __init__(self):
        self.scene_gate = SceneGate()
        self.mode = "single"
        self.last_active_cells = None
        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0

        self.internal_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
        self.current_piece_name = None
        self.current_rotation = 0
        self.current_col = 0
        self.current_row = 0
        self.current_cells = None
        self.queue_pieces_state = []
        self.needs_state_update = True

    def run(self, controller, tasker, mode="single"):
        self.mode = mode
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

    def _sleep_with_stop(self, tasker, seconds: float) -> bool:
        end_at = time.time() + seconds
        while time.time() < end_at:
            if tasker.stopping:
                return False
            time.sleep(min(0.1, end_at - time.time()))
        return True

    def _wait_for_next_piece(self, controller, tasker):
        import cv2
        board_h = BOARD_REGION[3]
        top_height = max(1, int(board_h * 2 / BOARD_ROWS))
        min_pixels = 20

        while True:
            if tasker.stopping:
                return False
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            board_crop = extract_board_crop(img)
            if board_crop is None or board_crop.size == 0:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            top_crop = board_crop[:top_height, :, :]
            if len(top_crop.shape) == 3 and top_crop.shape[2] == 4:
                top_crop = cv2.cvtColor(top_crop, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
            bright_pixels = int(np.count_nonzero(hsv[:, :, 2] >= REAL_BLOCK_VALUE_THRESHOLD))

            if bright_pixels >= min_pixels:
                print(f"[NewPiece] Block detected in top region ({bright_pixels} px), new piece spawned")
                return True

            if not self._sleep_with_stop(tasker, 0.05):
                return False

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

        grid = extract_visible_grid(board_crop, debug=DEBUG_BOARD)
        active_cells = identify_active_piece(grid, prefer_cells=self.last_active_cells)
        piece_state = match_piece_state(active_cells) if active_cells else None
        queue_pieces = self.scene_gate.read_piece_queue(img)

        if active_cells is not None:
            self.last_active_cells = active_cells

        return {
            "board_crop": board_crop,
            "grid": grid,
            "active_cells": active_cells,
            "piece_state": piece_state,
            "queue_pieces": queue_pieces,
        }

    def _update_game_state(self, img):
        board_crop = extract_board_crop(img)
        if board_crop is None or board_crop.size == 0:
            return False

        grid = extract_visible_grid(board_crop, debug=DEBUG_BOARD)
        active_cells = identify_active_piece(grid, prefer_cells=None)

        if active_cells is None:
            return False

        piece_state = match_piece_state(active_cells)
        if piece_state is None:
            return False

        self.internal_board = grid.copy()
        for row, col in active_cells:
            self.internal_board[row, col] = False

        self.current_piece_name = piece_state["piece"]
        self.current_rotation = piece_state["rotation"]
        self.current_col = piece_state["col"]
        self.current_row = piece_state["row"]
        self.current_cells = piece_state["cells"]

        self.queue_pieces_state = self.scene_gate.read_piece_queue(img)
        if self.queue_pieces_state:
            print(f"Queue(bottom->top)={self.queue_pieces_state}")

        self.last_active_cells = active_cells
        return True

    def _apply_move_no_feedback(self, controller, target_rotation, target_col):
        rotation_count = len(PIECES[self.current_piece_name])

        clockwise_steps = (target_rotation - self.current_rotation) % rotation_count
        counterclockwise_steps = (self.current_rotation - target_rotation) % rotation_count

        if clockwise_steps <= counterclockwise_steps:
            for _ in range(clockwise_steps):
                self._tap_key(controller, VK_K, hold=0.03)
                time.sleep(0.03)
                self.current_rotation = (self.current_rotation + 1) % rotation_count
        else:
            for _ in range(counterclockwise_steps):
                self._tap_key(controller, VK_J, hold=0.03)
                time.sleep(0.03)
                self.current_rotation = (self.current_rotation - 1) % rotation_count

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

    def _classify_scene(self, img, play_state=None):
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
            result = self._wait_for_scene_names(
                controller,
                tasker,
                {"game_active", "game_idle", "world_prompt", "world_no_prompt", "exit"},
                timeout_seconds=30.0,
            )
            if result is None:
                print("Loading wait timed out.")
                return False
            scene_name = result["name"]
            if scene_name in ("game_active", "game_idle"):
                return self._play_from_game(controller, tasker)
            if scene_name in ("world_prompt", "world_no_prompt"):
                scene = result

        if scene_name == "exit":
            print(f"Exit scene at start, clicking exit. score={scene['score']:.2f}")
            self._click_template(controller, scene["x"], scene["y"], scene["template"])
            if not self._wait_until_exit_to_world(controller, tasker):
                return False
            img = self._safe_get_image(controller)
            if img is None:
                return False
            scene = self._classify_scene(img)
            scene_name = scene["name"]

        if scene_name == "world_prompt":
            print(f"World with prompt, pressing F. score={scene['score']:.2f}")
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
            print(
                f"Found prepare_two at start, clicking return to re-select mode. score={scene['score']:.2f}"
            )
            img = self._safe_get_image(controller)
            if img is not None:
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
            print(f"{mode_label}模式入口, clicking. score={scene['score']:.2f}")
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
            print(f"Start-match scene, clicking. score={scene['score']:.2f}")
            self._click_point(controller, *PREPARE_TWO_CLICK_POINT)
            expected_after_start = {"game_active", "game_idle", "exit", "loading"}
            result = self._wait_for_scene_names(
                controller,
                tasker,
                expected_after_start,
                timeout_seconds=12.0,
            )
            if result is None:
                return False
            scene_name = result["name"]

            if scene_name == "loading":
                print("Loading after match, waiting...")
                result = self._wait_for_scene_names(
                    controller,
                    tasker,
                    {"game_active", "game_idle", "world_prompt", "world_no_prompt"},
                    timeout_seconds=30.0,
                )
                if result is None:
                    print("Loading wait timed out.")
                    return False
                scene_name = result["name"]

        if scene_name in ("game_active", "game_idle"):
            return self._play_from_game(controller, tasker)

        print(f"Unexpected scene after navigation: {scene_name}")
        return False

    _SCENE_LOCK_SEC = 3.0

    def _play_from_game(self, controller, tasker):
        last_piece_signature = None
        skip_count = 0
        round_start = time.time()
        non_active_since = None
        scene_lock_until = time.time() + self._SCENE_LOCK_SEC
        scene_locked = True

        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0
        self.needs_state_update = True
        self.current_piece_name = None

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

            if scene_locked and now < scene_lock_until:
                if self.current_piece_name is not None:
                    scene_lock_until = now + self._SCENE_LOCK_SEC
                elif play_state_for_scene is not None and play_state_for_scene["piece_state"] is not None:
                    scene_lock_until = now + self._SCENE_LOCK_SEC
                else:
                    scene_locked = False

            if not scene_locked:
                scene = self._classify_scene(img, play_state_for_scene)
                scene_name = scene["name"]

                if scene_name == "exit":
                    print(
                        f"Exit detected, clicking to leave match. score={scene['score']:.2f}"
                    )
                    self._click_template(
                        controller, scene["x"], scene["y"], self.scene_gate.exit_tpl
                    )
                    self._wait_until_exit_to_world(controller, tasker)
                    print("=== Auto Tetris Finished ===")
                    return True

                if scene_name == "game_idle":
                    self.last_active_cells = None
                    if play_state_for_scene is not None and play_state_for_scene["piece_state"] is not None:
                        scene_locked = True
                        scene_lock_until = now + self._SCENE_LOCK_SEC
                        self.needs_state_update = True
                    else:
                        if not self._sleep_with_stop(tasker, 0.15):
                            return False
                        continue

                elif scene_name == "loading":
                    self.last_active_cells = None
                    print("Loading detected during play, waiting...")
                    result = self._wait_for_scene_names(
                        controller,
                        tasker,
                        {"game_active", "game_idle", "world_prompt", "world_no_prompt"},
                        timeout_seconds=30.0,
                    )
                    if result is None:
                        return False
                    if result["name"] in ("game_active", "game_idle"):
                        scene_locked = True
                        scene_lock_until = time.time() + self._SCENE_LOCK_SEC
                        self.needs_state_update = True
                        continue
                    print(f"Unexpected scene after loading: {result['name']}")
                    return False

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
                        self.needs_state_update = True
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
                            self.needs_state_update = True
                            if not self._sleep_with_stop(tasker, 0.4):
                                return False
                            continue

                        if not self._recover_from_scene(controller, tasker, scene):
                            return False
                        continue

            if self.needs_state_update:
                if not self._update_game_state(img):
                    if not scene_locked:
                        if not self._sleep_with_stop(tasker, 0.08):
                            return False
                    else:
                        if not self._sleep_with_stop(tasker, 0.04):
                            return False
                    continue
                self.needs_state_update = False
                scene_locked = True
                scene_lock_until = time.time() + self._SCENE_LOCK_SEC

            if self.current_piece_name is None:
                if not self._sleep_with_stop(tasker, 0.04):
                    return False
                continue

            non_active_since = None

            current_signature = (self.current_piece_name, self.current_rotation, self.current_col)
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

            self._apply_move_no_feedback(
                controller, best_move["rotation"], best_move["target_col"]
            )

            self._wait_for_next_piece(controller, tasker)

            if best_move.get("lines_cleared", 0) > 0:
                now = time.time()
                if now - self.last_clear_time < 3.0:
                    self.combo_count += 1
                else:
                    self.combo_count = 1
                self.last_clear_time = now
                self.total_lines_cleared += best_move["lines_cleared"]

                if best_move.get("is_t_spin"):
                    print(f"[Special] T-SPIN detected! Combo={self.combo_count}")
                if self.combo_count > 1:
                    print(f"[Special] COMBO x{self.combo_count}!")
                print(f"[Stats] Total lines: {self.total_lines_cleared}")
            else:
                if time.time() - self.last_clear_time > 5.0:
                    self.combo_count = 0

            self.needs_state_update = True
            self.current_piece_name = None
            last_piece_signature = None

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        print("Tetris round timed out.")
        return False

    def _recover_from_scene(self, controller, tasker, scene: dict) -> bool:
        scene_name = scene["name"]
        if scene_name == "world_prompt":
            print(f"Recovery: world prompt, pressing F. score={scene['score']:.2f}")
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
            print(
                f"Recovery: {mode_label}模式入口, clicking. score={scene['score']:.2f}"
            )
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
            print(f"Recovery: start-match, clicking. score={scene['score']:.2f}")
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
            print(f"Recovery: exit button, clicking. score={scene['score']:.2f}")
            self._click_template(controller, scene["x"], scene["y"], scene["template"])
            return self._wait_until_exit_to_world(
                controller, tasker, timeout_seconds=6.0
            )

        if scene_name == "loading":
            print("Recovery: loading screen, waiting...")
            result = self._wait_for_scene_names(
                controller,
                tasker,
                {"game_active", "game_idle", "world_prompt", "world_no_prompt"},
                timeout_seconds=30.0,
            )
            return result is not None and result["name"] in ("game_active", "game_idle")

        if scene_name == "game_idle":
            return self._sleep_with_stop(tasker, 0.08)

        img = self._safe_get_image(controller)
        if img is not None:
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
                self._click_template(
                    controller, scene["x"], scene["y"], scene["template"]
                )
                if self._wait_until_exit_to_world(
                    controller, tasker, timeout_seconds=4.0
                ):
                    return True
                continue

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
                    t_spin_result = detect_t_spin(board, piece_name, rotation_index, target_col, result["row"])
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
                    eval_score = evaluate_board_fast(result["board"], result["lines_cleared"], combo_count=next_combo)

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
                    total_score = candidate["score"] + future_value * future_weight * depth_discount

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
                result = simulate_drop(board, shape, target_col)
                if result is None:
                    continue

                is_t_spin = False
                from ..utils.board import detect_t_spin
                if piece_name == "T":
                    t_spin_result = detect_t_spin(board, piece_name, rotation_index, target_col, result["row"])
                    is_t_spin = t_spin_result["is_t_spin"]

                future_bonus = 0.0
                if future_queue:
                    next_combo = self.combo_count + 1 if result["lines_cleared"] > 0 else 0
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
                    combo_count=self.combo_count + 1 if result["lines_cleared"] > 0 else 0,
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
