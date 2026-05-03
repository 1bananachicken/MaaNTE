import cv2
import time
import json
import numpy as np
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .Common.utils import get_image, match_template_in_region

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


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
    fallback = here.parents[4] / "resource" / "base" / "image" / "Tetris"
    return fallback


def _load_templates(subdir: str) -> list:
    image_root = _find_image_root()
    tpl_dir = image_root / subdir
    if not tpl_dir.is_dir():
        return []
    templates = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in _IMAGE_EXTS:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                img_bytes = np.fromfile(str(p), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                templates.append((p.stem, img))
    return templates


def _vote(img, region, templates, threshold) -> int:
    votes = 0
    for _, tpl in templates:
        matched, _, _, _ = match_template_in_region(img, region, tpl, threshold)
        if matched:
            votes += 1
    return votes


_LEFT_ROI = [24, 397, 340, 211]
_RIGHT_ROI = [965, 540, 230, 53]
_RESULT_ROI = [429, 220, 150, 81]


@AgentServer.custom_action("auto_tetris")
class AutoTetris(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        left_templates = _load_templates("left_region")
        right_templates = _load_templates("right_region")
        result_templates = _load_templates("result")

        vote_threshold = 1
        match_threshold = 0.7
        check_freq = 0.5
        playing_timeout = 30
        result_timeout = 300

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                vote_threshold = params.get("vote_threshold", 1)
                match_threshold = params.get("match_threshold", 0.7)
                check_freq = params.get("freq", 0.5)
                playing_timeout = params.get("playing_timeout", 30)
                result_timeout = params.get("result_timeout", 300)
            except Exception:
                pass

        print(
            f"=== Auto Tetris Started | "
            f"left_templates={len(left_templates)} "
            f"right_templates={len(right_templates)} "
            f"result_templates={len(result_templates)} "
            f"vote_threshold={vote_threshold} "
            f"match_threshold={match_threshold} ==="
        )

        playing_detected = False
        start_time = time.time()

        while time.time() - start_time < playing_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            img = get_image(controller)
            left_votes = _vote(img, _LEFT_ROI, left_templates, match_threshold)
            right_votes = _vote(img, _RIGHT_ROI, right_templates, match_threshold)

            if left_votes >= vote_threshold and right_votes >= vote_threshold:
                print(
                    f"Playing detected | left_votes={left_votes} right_votes={right_votes}"
                )
                playing_detected = True
                break

            time.sleep(check_freq)

        if not playing_detected:
            print("Auto Tetris: playing state not detected within timeout")
            return CustomAction.RunResult(success=False)

        start_time = time.time()

        while time.time() - start_time < result_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            img = get_image(controller)
            left_votes = _vote(img, _LEFT_ROI, left_templates, match_threshold)
            right_votes = _vote(img, _RIGHT_ROI, right_templates, match_threshold)
            result_votes = _vote(img, _RESULT_ROI, result_templates, match_threshold)

            if (
                left_votes >= vote_threshold
                and right_votes >= vote_threshold
                and result_votes >= vote_threshold
            ):
                print(
                    f"Result detected | left_votes={left_votes} "
                    f"right_votes={right_votes} result_votes={result_votes}"
                )
                return CustomAction.RunResult(success=True)

            time.sleep(check_freq)

        print("Auto Tetris: result not detected within timeout")
        return CustomAction.RunResult(success=False)