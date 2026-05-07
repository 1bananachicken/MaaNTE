from pathlib import Path

import cv2
import numpy as np

from ...Common.utils import match_template_in_region

DROP_BUTTON_REGION = [270, 425, 92, 97]
MATCHEND_REGION = [355, 11, 354, 122]

VK_A = 65
VK_D = 68
VK_J = 74
VK_K = 75


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
        self.drop_tpl = _read_image("drop.png")
        self.matchend_tpl = _read_image("Matchend.png")

        self.active_piece_templates = self._load_block_templates("blocks/active")
        self.queue_piece_templates = self._load_block_templates("blocks/queue")

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

    def _find_drop_button(self, img):
        if self.drop_tpl is None:
            return False, 0.0, 0, 0
        return self._match_template_region(
            img,
            DROP_BUTTON_REGION,
            self.drop_tpl,
            0.70,
        )

    def _find_matchend(self, img):
        if self.matchend_tpl is None:
            return False, 0.0, 0, 0
        return self._match_template_region(
            img,
            MATCHEND_REGION,
            self.matchend_tpl,
            0.75,
        )

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
