import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .utils import get_image, match_template_in_region, click_rect


abs_path = Path(__file__).parents[3]
if Path.exists(abs_path / "assets"):
    image_dir = abs_path / "assets/resource/base/image/auto_play_drums"
else:
    image_dir = abs_path / "resource/base/image/auto_play_drums"

TEMPLATE_START = cv2.imread(str(image_dir / "start_playing.png"), cv2.IMREAD_COLOR)
TEMPLATE_FINISH = cv2.imread(str(image_dir / "finish.png"), cv2.IMREAD_COLOR)

DRUM_KEYS = {
    "press_d.png": 68,
    "press_f.png": 70,
    "press_j.png": 74,
    "press_k.png": 75,
}
DRUM_KEY_NAMES = {v: k.replace("press_", "").replace(".png", "").upper() for k, v in DRUM_KEYS.items()}
DRUM_TEMPLATES = {name: cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR) for name in DRUM_KEYS}


@AgentServer.custom_action("auto_play_drums")
class AutoPlayDrums(CustomAction):
    def run(self, context: Context, argv) -> CustomAction.RunResult:
        controller = context.tasker.controller
        stopping = False

        screen_w, screen_h = 1280, 720
        start_region = (screen_w // 2, screen_h // 2, screen_w, screen_h)
        # 速度快的用上面这个
        # drum_regions = {
        #     "press_d.png": {"region": (255, 440, 335, 600), "threshold": 0.7},
        #     "press_f.png": {"region": (475, 450, 565, 600), "threshold": 0.7},
        #     "press_j.png": {"region": (710, 450, 800, 600), "threshold": 0.7},
        #     "press_k.png": {"region": (945, 440, 1025, 600), "threshold": 0.7},
        # }
        # 速度慢的用下面这个
        drum_regions = {
            "press_d.png": {"region": (255, 440, 335, 600), "threshold": 0.81},
            "press_f.png": {"region": (475, 450, 565, 600), "threshold": 0.80},
            "press_j.png": {"region": (710, 450, 800, 600), "threshold": 0.80},
            "press_k.png": {"region": (945, 440, 1025, 600), "threshold": 0.81},
        }

        # 阶段1: 等待并点击开始
        while not stopping:
            stopping = context.tasker.stopping
            time.sleep(0.1)
            img = get_image(controller)
            if img is None or stopping:
                continue

            matched, _, x, y = match_template_in_region(img, start_region, TEMPLATE_START, min_similarity=0.8)
            if matched:
                click_rect(controller, (x, y, TEMPLATE_START.shape[1], TEMPLATE_START.shape[0]))
                break

        time.sleep(3)

        # 阶段2: 游戏循环 - 持续检测鼓点
        with ThreadPoolExecutor(max_workers=5) as executor:
            while not stopping:
                stopping = context.tasker.stopping
                img = get_image(controller)
                if img is None or stopping:
                    continue

                futures = {}
                for name, template in DRUM_TEMPLATES.items():
                    cfg = drum_regions[name]
                    futures[executor.submit(match_template_in_region, img, cfg["region"], template, cfg["threshold"])] = name

                matched_keys = []
                for future in futures:
                    name = futures[future]
                    m, _, _, _ = future.result()
                    if m:
                        matched_keys.append(DRUM_KEYS[name])

                # 检测 finish
                m_finish, _, _, _ = match_template_in_region(img, (540, 620, 740, 670), TEMPLATE_FINISH, min_similarity=0.8)
                if m_finish:
                    print("=== Detected finish, pressing ESC ===")
                    controller.post_key_down(27)
                    time.sleep(0.1)
                    controller.post_key_up(27)
                    break

                if not matched_keys:
                    continue
                print(f"=== Press {[DRUM_KEY_NAMES[k] for k in matched_keys]} ===")

                for key in matched_keys:
                    controller.post_key_down(key)
                time.sleep(0.01)
                for key in matched_keys:
                    controller.post_key_up(key)

        return CustomAction.RunResult(success=True)
