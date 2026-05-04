import time

import numpy as np

from ...Common.utils import get_image
from ..utils.board import (
    BOARD_COLS,
    BOARD_ROWS,
    DEBUG_BOARD,
    extract_board_crop,
    extract_visible_grid,
    identify_active_piece,
    simulate_drop,
    evaluate_board,
    dump_board_state,
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
        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0

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

    def _wait_for_new_piece(self, controller, tasker, old_piece_name: str, timeout: float = 1.2):
        start = time.time()
        while time.time() - start < timeout:
            if tasker.stopping:
                return False
            img = self._safe_get_image(controller)
            if img is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue
            play_state = self._scan_play_state(img)
            if play_state is None or play_state["piece_state"] is None:
                if not self._sleep_with_stop(tasker, 0.05):
                    return False
                continue
            new_piece = play_state["piece_state"]["piece"]
            if new_piece != old_piece_name:
                print(f"[Landing] New piece detected: {new_piece} (old: {old_piece_name})")
                return True
            if not self._sleep_with_stop(tasker, 0.05):
                return False
        print(f"[Landing] Wait for new piece timed out after {timeout}s")
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

    def _diagnose_move_failure(self, settled_board, piece_state, best_move, attempt_num):
        """诊断移动失败原因，输出详细调试信息"""
        piece_name = piece_state["piece"]
        target_rotation = best_move["rotation"]
        target_col = best_move["target_col"]
        current_col = piece_state["col"]
        current_rotation = piece_state["rotation"]

        print(f"[MoveDiag] === 诊断开始 (attempt #{attempt_num}) ===")
        print(f"[MoveDiag] Piece={piece_name} current_rot={current_rotation} current_col={current_col}")
        print(f"[MoveDiag] Target: rot={target_rotation} col={target_col}")

        # 检查每个旋转状态的 drop 结果
        from ..utils.pieces import PIECES
        shapes = PIECES[piece_name]
        for rot_idx, shape in enumerate(shapes):
            width = max(col for _, col in shape) + 1
            for tcol in range(0, BOARD_COLS - width + 1):
                result = simulate_drop(settled_board, shape, tcol)
                if result is None:
                    # 诊断碰撞原因 - 使用 simulate_drop 的逻辑检查实际落点
                    collision_cells = []
                    shape_height = max(row for row, _ in shape) + 1
                    # 找到实际会碰撞的行
                    for test_row in range(BOARD_ROWS):
                        collides = False
                        for row_offset, col_offset in shape:
                            r = test_row + row_offset
                            c = tcol + col_offset
                            if r >= BOARD_ROWS or c < 0 or c >= BOARD_COLS:
                                collides = True
                                collision_cells.append(f"({r},{c})OUT_OF_BOUNDS")
                                break
                            if r >= 0 and settled_board[r, c]:
                                collides = True
                                collision_cells.append(f"({r},{c})BLOCKED")
                                break
                        if collides:
                            break
                    if rot_idx == target_rotation and tcol == target_col:
                        print(f"[MoveDiag] TARGET rot={rot_idx} col={tcol}: COLLISION at row~{test_row} {collision_cells}")
                elif rot_idx == target_rotation and tcol == target_col:
                    print(f"[MoveDiag] TARGET rot={rot_idx} col={tcol}: OK score={result['score']:.2f} land_row={result['row']}")

        # dump 棋盘状态
        board_content = dump_board_state(
            settled_board,
            active_cells=piece_state["cells"],
            piece_state=piece_state,
            filepath="tetris_debug_move_fail.txt",
        )
        print(f"[MoveDiag] Board state:\n{board_content}")
        print(f"[MoveDiag] === 诊断结束 ===")

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
        move_fail_count = 0
        round_start = time.time()
        non_active_since = None
        scene_lock_until = time.time() + self._SCENE_LOCK_SEC
        scene_locked = True
        last_piece_name = None

        self.combo_count = 0
        self.last_clear_time = 0
        self.total_lines_cleared = 0
        self.last_hard_drop_time = 0

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
                    scene_lock_until = now + self._SCENE_LOCK_SEC
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
                        scene_lock_until = now + self._SCENE_LOCK_SEC
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

            current_signature = (piece_state["piece"], piece_state["rotation"], piece_state["col"])
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
            scene_locked = True
            scene_lock_until = time.time() + self._SCENE_LOCK_SEC

            decision_start = time.time()

            settled_board = grid.copy()
            for row, col in active_cells:
                settled_board[row, col] = False

            if queue_pieces:
                print(f"Queue(bottom->top)={queue_pieces}")

            if queue_pieces:
                planning_queue = [piece_state["piece"], *queue_pieces[:5]]
            else:
                planning_queue = [piece_state["piece"]]

            planning_queue = planning_queue[:6]

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

            decision_time = time.time() - decision_start
            if decision_time > 0.3:
                print(f"[Perf] Decision took {decision_time:.3f}s")

            now = time.time()
            time_since_last_drop = now - self.last_hard_drop_time
            if time_since_last_drop < 0.8:
                wait_time = 0.8 - time_since_last_drop
                print(f"[RateLimit] Waiting {wait_time:.2f}s before next hard drop")
                if not self._sleep_with_stop(tasker, wait_time):
                    return False

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
                last_piece_name = piece_state["piece"]
                self._wait_for_new_piece(controller, tasker, last_piece_name)
                self.last_hard_drop_time = time.time()
                last_piece_signature = (piece_state["piece"], piece_state["rotation"], piece_state["col"])
                move_fail_count = 0

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
            else:
                move_fail_count += 1
                print(
                    "Move did not reach target position, retrying with a fresh board state."
                )
                # 连续失败3次时输出诊断信息
                if move_fail_count >= 3:
                    self._diagnose_move_failure(
                        settled_board, piece_state, best_move, move_fail_count
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

    def _apply_move_with_feedback(self, controller, tasker, piece_state, best_move):
        piece_name = piece_state["piece"]
        rotation_count = len(PIECES[piece_name])
        target_rotation = best_move["rotation"]
        target_col = best_move["target_col"]
        deadline = time.time() + 3.0
        hard_drop_sent = False
        hard_drop_method = None  # "space" or "button"
        hard_drop_deadline = None  # hard drop 阶段的独立超时
        post_drop_deadline = None
        space_drop_attempt_time = None

        expected_rot_change = False
        expected_col_change = False
        last_rotation = piece_state["rotation"]
        last_col = piece_state["col"]
        last_action_time = 0.0

        FEEDBACK_TIMEOUT = 0.50  # 反馈超时时间

        while time.time() < deadline or (hard_drop_sent and time.time() < hard_drop_deadline):
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
                # SPACE 发送后方块仍在 → 尝试点击 drop 按钮
                if hard_drop_method == "space" and space_drop_attempt_time is not None:
                    if time.time() - space_drop_attempt_time >= 0.50:
                        print("[KeyInput] SPACE drop failed, trying drop button click")
                        found, score, dx, dy = self.scene_gate._find_drop_button(img)
                        if found:
                            click_x = int(dx + self.scene_gate.drop_tpl.shape[1] / 2)
                            click_y = int(dy + self.scene_gate.drop_tpl.shape[0] / 2)
                            self._click_point(controller, click_x, click_y)
                            print(f"[KeyInput] Drop button clicked at ({click_x},{click_y}) score={score:.2f}")
                            hard_drop_method = "button"
                            space_drop_attempt_time = None
                            post_drop_deadline = time.time() + 0.9
                            if not self._sleep_with_stop(tasker, 0.10):
                                return False
                            continue
                        else:
                            print("[KeyInput] Drop button not found, retrying SPACE")
                            self._tap_key(controller, VK_SPACE, hold=0.06)
                            space_drop_attempt_time = time.time()
                            if not self._sleep_with_stop(tasker, 0.08):
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
                if time.time() - last_action_time < FEEDBACK_TIMEOUT:
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                    continue
                print(f"[KeyFeedback] ROTATION FAILED: still at rot={current_rotation} after {FEEDBACK_TIMEOUT}s")
                return False

            if expected_col_change and current_col == last_col:
                if time.time() - last_action_time < FEEDBACK_TIMEOUT:
                    if not self._sleep_with_stop(tasker, 0.04):
                        return False
                    continue
                print(f"[KeyFeedback] MOVEMENT FAILED: still at col={current_col} after {FEEDBACK_TIMEOUT}s (expected col change)")
                return False

            expected_rot_change = False
            expected_col_change = False
            last_rotation = current_rotation
            last_col = current_col

            if current_rotation == target_rotation and current_col == target_col:
                if not hard_drop_sent:
                    print(f"[KeyInput] HARD DROP: piece at target rot={current_rotation} col={current_col}")
                    self._tap_key(controller, VK_SPACE, hold=0.06)
                    hard_drop_sent = True
                    hard_drop_method = "space"
                    hard_drop_deadline = time.time() + 3.0
                    space_drop_attempt_time = time.time()
                    post_drop_deadline = time.time() + 1.5
                    if not self._sleep_with_stop(tasker, 0.10):
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
                key_name = "K(clockwise)" if correction_key == VK_K else "J(counter)"
                print(f"[KeyInput] ROTATE: {key_name} current_rot={current_rotation} -> target_rot={target_rotation}")
                self._tap_key(controller, correction_key, hold=0.05)
                last_action_time = time.time()
                expected_rot_change = True
                if not self._sleep_with_stop(tasker, 0.08):
                    return False
                continue

            if current_col != target_col:
                if (current_col == 0 and target_col < current_col) or (
                    current_col == BOARD_COLS - 1 and target_col > current_col
                ):
                    print(f"[KeyInput] BOUNDARY BLOCKED: current_col={current_col} target_col={target_col} BOARD_COLS={BOARD_COLS}")
                    return False
                correction_key = VK_D if current_col < target_col else VK_A
                key_name = "D(right)" if correction_key == VK_D else "A(left)"
                print(f"[KeyInput] MOVE: {key_name} current_col={current_col} -> target_col={target_col}")
                self._tap_key(controller, correction_key, hold=0.05)
                last_action_time = time.time()
                expected_col_change = True
                if not self._sleep_with_stop(tasker, 0.08):
                    return False
                continue

            if not self._sleep_with_stop(tasker, 0.04):
                return False

        return False
