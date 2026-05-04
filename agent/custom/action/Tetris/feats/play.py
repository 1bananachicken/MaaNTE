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
    VK_S,
    VK_SPACE,
)


class TetrisGamePlayer:
    def __init__(self):
        self.scene_gate = SceneGate()
        self.mode = "single"
        self.last_active_cells = None

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
        scene_lock_until = time.time() + _SCENE_LOCK_SEC
        scene_locked = True

        while time.time() - round_start < 900:
            if tasker.stopping:
                return False

            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue

            now = time.time()
            play_state = self._scan_play_state(img)

            if scene_locked and now < scene_lock_until:
                if play_state is not None and play_state["piece_state"] is not None:
                    scene_lock_until = now + _SCENE_LOCK_SEC
                else:
                    scene_locked = False

            if not scene_locked:
                scene = self._classify_scene(img, play_state)
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
                    if play_state is not None and play_state["piece_state"] is not None:
                        scene_locked = True
                        scene_lock_until = now + _SCENE_LOCK_SEC
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
                        scene_lock_until = time.time() + _SCENE_LOCK_SEC
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
                        and play_state is not None
                        and play_state["piece_state"] is not None
                    ):
                        self.last_active_cells = None
                        scene_locked = True
                        scene_lock_until = now + _SCENE_LOCK_SEC
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

            if play_state is None or play_state["piece_state"] is None:
                if not scene_locked:
                    if not self._sleep_with_stop(tasker, 0.08):
                        return False
                else:
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                continue

            non_active_since = None

            grid = play_state["grid"]
            active_cells = play_state["active_cells"]
            piece_state = play_state["piece_state"]
            queue_pieces = play_state["queue_pieces"]

            if piece_state["cells"] == last_piece_signature:
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
            scene_locked = True
            scene_lock_until = time.time() + _SCENE_LOCK_SEC

            settled_board = grid.copy()
            for row, col in active_cells:
                settled_board[row, col] = False

            if queue_pieces:
                print(f"Queue(bottom->top)={queue_pieces}")

            if queue_pieces:
                planning_queue = [piece_state["piece"], *queue_pieces[:3]]
            else:
                planning_queue = [piece_state["piece"]]

            planning_queue = planning_queue[:4]

            best_move = self._choose_best_current_piece_move(
                settled_board,
                piece_state,
                planning_queue,
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
            move_applied = self._apply_move_with_feedback(
                controller, tasker, piece_state, best_move
            )
            if move_applied:
                last_piece_signature = piece_state["cells"]
            else:
                print(
                    "Move did not reach target position, retrying with a fresh board state."
                )
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
        max_depth=4,
        beam_width=8,
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
                candidates.append(
                    {
                        "rotation": rotation_index,
                        "target_col": target_col,
                        "score": result["score"],
                        "board": result["board"],
                        "piece": piece_name,
                    }
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)
        search_candidates = candidates[:beam_width]

        best_choice = None
        for candidate in search_candidates:
            total_score = candidate["score"]
            if depth + 1 < max_depth and len(queue_pieces) > 1:
                future = self._search_best_queue_move(
                    candidate["board"],
                    queue_pieces[1:],
                    depth=depth + 1,
                    max_depth=max_depth,
                    beam_width=beam_width,
                )
                if future is not None:
                    # Only evaluate terminal state value, don't accumulate intermediate board evaluations
                    # However, we might want to still reward early lines cleared, but since `score` represents 
                    # board evaluation, propagating the max future score is structurally more sound for a state value function.
                    total_score = future["total_score"]

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
                        result["board"],
                        future_queue,
                        max_depth=min(4, len(future_queue)),
                        beam_width=8,
                    )
                    if future_move is not None:
                        future_bonus = future_move["total_score"] * 0.38

                rot_dist = rotation_distance(
                    piece_name, piece_state["rotation"], rotation_index
                )
                shift_distance = abs(target_col - piece_state["col"])
                execution_penalty = rot_dist * 0.5 + shift_distance * 0.12
                if piece_name == "I" and rot_dist > 0:
                    execution_penalty += 0.3
                if rot_dist > 0 and shift_distance > 3:
                    execution_penalty += 0.2

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

        expected_rot_change = False
        expected_col_change = False
        last_rotation = piece_state["rotation"]
        last_col = piece_state["col"]
        last_action_time = 0.0

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
                    if (
                        post_drop_deadline is not None
                        and time.time() >= post_drop_deadline
                    ):
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

            if expected_rot_change and current_rotation == last_rotation:
                if time.time() - last_action_time < 0.20:
                    if not self._sleep_with_stop(tasker, 0.03):
                        return False
                    continue
                return False

            if expected_col_change and current_col == last_col:
                if time.time() - last_action_time < 0.20:
                    if not self._sleep_with_stop(tasker, 0.03):
                        return False
                    continue
                return False

            expected_rot_change = False
            expected_col_change = False
            last_rotation = current_rotation
            last_col = current_col

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
            counterclockwise_steps = (
                current_rotation - target_rotation
            ) % rotation_count

            if current_rotation != target_rotation:
                correction_key = (
                    VK_K if clockwise_steps <= counterclockwise_steps else VK_J
                )
                self._tap_key(controller, correction_key, hold=0.04)
                last_action_time = time.time()
                expected_rot_change = True
                if not self._sleep_with_stop(tasker, 0.03):
                    return False
                continue

            if current_col != target_col:
                if (current_col == 0 and target_col < current_col) or (
                    current_col == BOARD_COLS - 1 and target_col > current_col
                ):
                    return False
                correction_key = VK_D if current_col < target_col else VK_A
                self._tap_key(controller, correction_key, hold=0.04)
                last_action_time = time.time()
                expected_col_change = True
                if not self._sleep_with_stop(tasker, 0.03):
                    return False
                continue

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        return False
