import time
from pathlib import Path

import cv2
import numpy as np

from ...Common.utils import get_image, match_template_in_region

WORLD_SCENE_MARKER_REGION = [174, 8, 55, 72]
PREPARE_ONE_SCENE_MARKER_REGION = [811, 149, 419, 168]
PREPARE_TWO_SCENE_MARKER_REGION = [809, 142, 422, 499]
GAME_SCENE_MARKER_REGION = [17, 90, 177, 79]
WORLD_TO_PREPARE_REGION = [775, 359, 94, 69]
WORLD_TO_PREPARE_FALLBACK_REGION = [700, 315, 240, 170]
PREPARE_ONE_REGION = [831, 176, 371, 136]
PREPARE_TWO_REGION = [1029, 650, 184, 48]
EXIT_REGION = [536, 601, 197, 39]
MULTIPLE_REGION = [825, 39, 130, 130]
LOADING_REGION = [1167, 638, 100, 70]
RETURN_REGION = [1195, 16, 50, 50]

PREPARE_ONE_CLICK_POINT = (1016, 244)
PREPARE_ONE_MULTI_CLICK_POINT = (885, 418)
PREPARE_TWO_CLICK_POINT = (1121, 674)

VK_A = 65
VK_D = 68
VK_F = 70
VK_J = 74
VK_K = 75
VK_ESC = 27
VK_S = 83
VK_SPACE = 32
VK_W = 87


DROP_BUTTON_REGION = [270, 425, 92, 97]


def _find_image_root() -> Path:
    here = Path(__file__).resolve()
    for i in range(len(here.parents)):
        root = here.parents[i]
        p1 = root / "resource" / "base" / "image" / "Tetris"
        if p1.is_dir():
            return p1
        p2 = root / "assets" / "resource" / "base" / "image" / "Tetris"
        if p2.is_dir():
            return p2
    fallback = here.parents[6] / "resource" / "base" / "image" / "Tetris"
    return fallback


_image_root = None


def get_image_root() -> Path:
    global _image_root
    if _image_root is None:
        _image_root = _find_image_root()
    return _image_root


def _read_image(name: str):
    path = get_image_root() / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        img_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img


class SceneGate:
    def __init__(self):
        self.world_scene_marker = _read_image("world_scene_marker.png")
        self.prepare_one_marker = _read_image("prepare_one_marker.png")
        self.prepare_two_marker = _read_image("prepare_two_marker.png")
        self.game_scene_marker = _read_image("game_scene_marker.png")
        self.world_prompt_tpl = _read_image("world_prompt.png")
        self.single_player_tpl = _read_image("single_player.png")
        self.multi_player_tpl = _read_image("multiple.png")
        self.start_match_tpl = _read_image("start_match.png")
        self.exit_tpl = _read_image("exit_button.png")
        self.loading_tpl = _read_image("loading.png")
        self.return_tpl = _read_image("return.png")
        self.drop_tpl = _read_image("drop.png")

        self.active_piece_templates = self._load_block_templates("blocks/active")
        self.queue_piece_templates = self._load_block_templates("blocks/queue")

        self.queue_preview_templates = {
            "T": self._make_preview_template(((0, 1), (1, 0), (1, 1), (1, 2))),
            "S": self._make_preview_template(((0, 1), (0, 2), (1, 0), (1, 1))),
            "Z": self._make_preview_template(((0, 0), (0, 1), (1, 1), (1, 2))),
            "J": self._make_preview_template(((0, 0), (1, 0), (1, 1), (1, 2))),
            "L": self._make_preview_template(((0, 2), (1, 0), (1, 1), (1, 2))),
            "O": self._make_preview_template(((0, 0), (0, 1), (1, 0), (1, 1))),
            "I": self._make_preview_template(((0, 0), (0, 1), (0, 2), (0, 3))),
        }

    def _make_preview_template(self, shape):
        rows = max(row for row, _ in shape) + 1
        cols = max(col for _, col in shape) + 1
        canvas = np.zeros((rows, cols), dtype=np.uint8)
        for row, col in shape:
            canvas[row, col] = 255
        return canvas

    def _load_block_templates(self, subdir: str):
        templates = {}
        base = get_image_root() / subdir
        for name in ("I", "J", "L", "O", "S", "T", "Z"):
            path = base / f"{name}.png"
            if path.is_file():
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    img_bytes = np.fromfile(str(path), dtype=np.uint8)
                    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    templates[name] = img
        return templates

    def match_active_piece(self, board_crop, min_similarity: float = 0.74):
        if board_crop is None or board_crop.size == 0:
            return None
        if not self.active_piece_templates:
            return None

        if len(board_crop.shape) == 3 and board_crop.shape[2] == 4:
            board_crop = cv2.cvtColor(board_crop, cv2.COLOR_BGRA2BGR)

        best = None
        board_gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)

        for piece_name, template in self.active_piece_templates.items():
            tpl = template
            if len(tpl.shape) == 3 and tpl.shape[2] == 4:
                tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

            if (
                board_gray.shape[0] < tpl_gray.shape[0]
                or board_gray.shape[1] < tpl_gray.shape[1]
            ):
                continue

            res = cv2.matchTemplate(board_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < min_similarity:
                continue

            if best is None or max_val > best["score"]:
                best = {
                    "piece": piece_name,
                    "score": float(max_val),
                    "x": int(max_loc[0]),
                    "y": int(max_loc[1]),
                    "w": tpl_gray.shape[1],
                    "h": tpl_gray.shape[0],
                }

        return best

    def match_active_piece_in_region(self, img, region, min_similarity: float = 0.74):
        if img is None or not isinstance(img, np.ndarray):
            return None

        x1, y1, width, height = region
        x2, y2 = x1 + width, y1 + height
        img_height, img_width = img.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = img[y1:y2, x1:x2]
        match = self.match_active_piece(roi, min_similarity=min_similarity)
        if match is None:
            return None

        match["x"] += x1
        match["y"] += y1
        return match

    def _match_template_region(
        self, img, region, template, min_similarity=0.8, grayscale=False
    ):
        if template is None:
            return False, 0.0, 0, 0

        if not grayscale:
            return match_template_in_region(img, region, template, min_similarity)

        if img is None or not isinstance(img, np.ndarray):
            return False, 0.0, 0, 0

        x1, y1, width, height = region
        x2, y2 = x1 + width, y1 + height
        img_height, img_width = img.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0, 0, 0

        roi = img[y1:y2, x1:x2]
        if len(roi.shape) == 3 and roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

        template_img = template
        if len(template_img.shape) == 3 and template_img.shape[2] == 4:
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2BGR)

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        template_gray = (
            cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            if len(template_img.shape) == 3
            else template_img
        )

        if (
            roi_gray.shape[0] < template_gray.shape[0]
            or roi_gray.shape[1] < template_gray.shape[1]
        ):
            return False, 0.0, 0, 0

        res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= min_similarity:
            return True, max_val, x1 + max_loc[0], y1 + max_loc[1]
        return False, max_val, 0, 0

    def _match_scene_marker(
        self, img, region, template, attempts=((0.78, False), (0.74, True))
    ):
        for threshold, grayscale in attempts:
            matched, score, x, y = self._match_template_region(
                img,
                region,
                template,
                threshold,
                grayscale=grayscale,
            )
            if matched:
                return True, score, x, y
        return False, 0.0, 0, 0

    def _find_world_prompt(self, img):
        matched, score, x, y = self._match_scene_marker(
            img,
            WORLD_SCENE_MARKER_REGION,
            self.world_scene_marker,
        )
        if matched:
            return True, score, x, y

        for region, threshold, grayscale in (
            (WORLD_TO_PREPARE_REGION, 0.75, False),
            (WORLD_TO_PREPARE_FALLBACK_REGION, 0.73, False),
            (WORLD_TO_PREPARE_FALLBACK_REGION, 0.71, True),
        ):
            matched, score, x, y = self._match_template_region(
                img,
                region,
                self.world_prompt_tpl,
                threshold,
                grayscale=grayscale,
            )
            if matched:
                return True, score, x, y

        return False, 0.0, 0, 0

    def _find_prepare_one(self, img):
        matched, score, x, y = self._match_scene_marker(
            img,
            PREPARE_ONE_SCENE_MARKER_REGION,
            self.prepare_one_marker,
        )
        if matched:
            return True, score, x, y, self.prepare_one_marker

        matched, score, x, y = match_template_in_region(
            img,
            PREPARE_ONE_REGION,
            self.single_player_tpl,
            0.75,
        )
        if matched:
            return True, score, x, y, self.single_player_tpl

        matched, score, x, y = match_template_in_region(
            img,
            MULTIPLE_REGION,
            self.multi_player_tpl,
            0.75,
        )
        if matched:
            return True, score, x, y, self.multi_player_tpl

        return False, 0.0, 0, 0, None

    def _find_prepare_two(self, img):
        matched, score, x, y = self._match_scene_marker(
            img,
            PREPARE_TWO_SCENE_MARKER_REGION,
            self.prepare_two_marker,
        )
        if matched:
            return True, score, x, y, self.prepare_two_marker

        matched, score, x, y = match_template_in_region(
            img,
            PREPARE_TWO_REGION,
            self.start_match_tpl,
            0.75,
        )
        if matched:
            return True, score, x, y, self.start_match_tpl

        return False, 0.0, 0, 0, None

    def _find_game_scene_marker(self, img):
        return self._match_scene_marker(
            img,
            GAME_SCENE_MARKER_REGION,
            self.game_scene_marker,
            attempts=((0.76, False), (0.72, True)),
        )

    def _find_loading(self, img):
        return self._match_scene_marker(
            img,
            LOADING_REGION,
            self.loading_tpl,
            attempts=((0.78, False), (0.74, True)),
        )

    def _find_return_button(self, img):
        if self.return_tpl is None:
            return False, 0.0, 0, 0
        return self._match_template_region(
            img,
            RETURN_REGION,
            self.return_tpl,
            0.72,
        )

    def _find_drop_button(self, img):
        if self.drop_tpl is None:
            return False, 0.0, 0, 0
        return self._match_template_region(
            img,
            DROP_BUTTON_REGION,
            self.drop_tpl,
            0.70,
        )

    def classify_scene(self, img, play_state=None):
        from ..utils.board import looks_like_game_scene

        if img is None or not isinstance(img, np.ndarray):
            return {
                "name": "unknown",
                "score": 0.0,
                "x": 0,
                "y": 0,
                "template": None,
                "play_state": play_state,
            }

        matched, score, x, y = match_template_in_region(
            img,
            EXIT_REGION,
            self.exit_tpl,
            0.75,
        )
        if matched:
            return {
                "name": "exit",
                "score": score,
                "x": x,
                "y": y,
                "template": self.exit_tpl,
                "play_state": play_state,
            }

        matched, score, x, y = self._find_loading(img)
        if matched:
            return {
                "name": "loading",
                "score": score,
                "x": x,
                "y": y,
                "template": self.loading_tpl,
                "play_state": play_state,
            }

        marker_matched, marker_score, marker_x, marker_y = self._match_scene_marker(
            img,
            WORLD_SCENE_MARKER_REGION,
            self.world_scene_marker,
        )
        prompt_matched = False
        prompt_score = 0.0
        prompt_x = 0
        prompt_y = 0
        if marker_matched:
            for region, threshold, grayscale in (
                (WORLD_TO_PREPARE_REGION, 0.75, False),
                (WORLD_TO_PREPARE_FALLBACK_REGION, 0.73, False),
                (WORLD_TO_PREPARE_FALLBACK_REGION, 0.71, True),
            ):
                pm, ps, px, py = self._match_template_region(
                    img,
                    region,
                    self.world_prompt_tpl,
                    threshold,
                    grayscale=grayscale,
                )
                if pm:
                    prompt_matched = True
                    prompt_score = ps
                    prompt_x = px
                    prompt_y = py
                    break
            if prompt_matched:
                return {
                    "name": "world_prompt",
                    "score": prompt_score,
                    "x": prompt_x,
                    "y": prompt_y,
                    "template": self.world_prompt_tpl,
                    "play_state": play_state,
                }
            return {
                "name": "world_no_prompt",
                "score": marker_score,
                "x": marker_x,
                "y": marker_y,
                "template": self.world_scene_marker,
                "play_state": play_state,
            }

        if not marker_matched:
            for region, threshold, grayscale in (
                (WORLD_TO_PREPARE_REGION, 0.75, False),
                (WORLD_TO_PREPARE_FALLBACK_REGION, 0.73, False),
                (WORLD_TO_PREPARE_FALLBACK_REGION, 0.71, True),
            ):
                pm, ps, px, py = self._match_template_region(
                    img,
                    region,
                    self.world_prompt_tpl,
                    threshold,
                    grayscale=grayscale,
                )
                if pm:
                    return {
                        "name": "world_prompt",
                        "score": ps,
                        "x": px,
                        "y": py,
                        "template": self.world_prompt_tpl,
                        "play_state": play_state,
                    }

        matched, score, x, y, template = self._find_prepare_two(img)
        if matched:
            return {
                "name": "prepare_two",
                "score": score,
                "x": x,
                "y": y,
                "template": template,
                "play_state": play_state,
            }

        matched, score, x, y, template = self._find_prepare_one(img)
        if matched:
            return {
                "name": "prepare_one",
                "score": score,
                "x": x,
                "y": y,
                "template": template,
                "play_state": play_state,
            }

        if play_state is not None and play_state.get("piece_state") is not None:
            return {
                "name": "game_active",
                "score": 1.0,
                "x": 0,
                "y": 0,
                "template": None,
                "play_state": play_state,
            }

        matched, score, x, y = self._find_game_scene_marker(img)
        if matched:
            return {
                "name": "game_idle",
                "score": score,
                "x": x,
                "y": y,
                "template": self.game_scene_marker,
                "play_state": play_state,
            }

        if looks_like_game_scene(img, play_state):
            return {
                "name": "game_idle",
                "score": 1.0,
                "x": 0,
                "y": 0,
                "template": None,
                "play_state": play_state,
            }

        return {
            "name": "unknown",
            "score": 0.0,
            "x": 0,
            "y": 0,
            "template": None,
            "play_state": play_state,
        }

    def read_piece_queue(self, img):
        from ..utils.board import extract_queue_crop

        queue_crop = extract_queue_crop(img)
        if queue_crop is None or queue_crop.size == 0:
            return []
        if not self.queue_piece_templates:
            return []

        if len(queue_crop.shape) == 3 and queue_crop.shape[2] == 4:
            queue_crop = cv2.cvtColor(queue_crop, cv2.COLOR_BGRA2BGR)

        queue_gray = cv2.cvtColor(queue_crop, cv2.COLOR_BGR2GRAY)
        candidates = []

        for piece_name, template in self.queue_piece_templates.items():
            tpl = template
            if len(tpl.shape) == 3 and tpl.shape[2] == 4:
                tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

            if (
                queue_gray.shape[0] < tpl_gray.shape[0]
                or queue_gray.shape[1] < tpl_gray.shape[1]
            ):
                continue

            res = cv2.matchTemplate(queue_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.74)
            for pt in zip(loc[1], loc[0]):
                score = float(res[pt[1], pt[0]])
                candidates.append(
                    {
                        "piece": piece_name,
                        "score": score,
                        "x": int(pt[0]),
                        "y": int(pt[1]),
                        "w": tpl_gray.shape[1],
                        "h": tpl_gray.shape[0],
                    }
                )

        if not candidates:
            return []

        candidates.sort(key=lambda item: item["score"], reverse=True)
        picked = []
        for cand in candidates:
            overlap = False
            for kept in picked:
                x1 = max(cand["x"], kept["x"])
                y1 = max(cand["y"], kept["y"])
                x2 = min(cand["x"] + cand["w"], kept["x"] + kept["w"])
                y2 = min(cand["y"] + cand["h"], kept["y"] + kept["h"])
                if x2 <= x1 or y2 <= y1:
                    continue
                inter = (x2 - x1) * (y2 - y1)
                area = cand["w"] * cand["h"]
                if inter / max(1, area) > 0.3:
                    overlap = True
                    break
            if not overlap:
                picked.append(cand)

        picked.sort(key=lambda item: item["y"], reverse=True)
        return [item["piece"] for item in picked[:6]]

    def _classify_queue_component(self, component_mask: np.ndarray, area: int = 0):
        height, width = component_mask.shape[:2]
        aspect_ratio = width / max(height, 1)

        if width >= 50 and height <= 22:
            return "I"
        fill_ratio = area / (width * height) if width * height > 0 else 0
        if (
            abs(width - height) <= 6
            and 22 <= width <= 34
            and 22 <= height <= 34
            and fill_ratio > 0.72
        ):
            return "O"

        best_piece = None
        best_iou = 0.0
        component_binary = component_mask > 0
        for piece_name, template in self.queue_preview_templates.items():
            resized_template = cv2.resize(
                template,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            template_binary = resized_template > 0
            intersection = np.logical_and(component_binary, template_binary).sum()
            union = np.logical_or(component_binary, template_binary).sum()
            if union == 0:
                continue
            iou = float(intersection / union)
            if iou > best_iou:
                best_iou = iou
                best_piece = piece_name

        if best_piece is not None and best_iou >= 0.55:
            return best_piece
        if best_piece is not None and best_iou >= 0.40 and aspect_ratio > 1.4:
            return best_piece
        return None
