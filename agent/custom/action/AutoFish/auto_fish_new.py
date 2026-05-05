import cv2
import time

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region
from ..Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = get_logger("auto_fish_new")


@AgentServer.custom_action("auto_fish_new")
class AutoFishNew(CustomAction):
    abs_path = Path(__file__).parents[4]
    if Path.exists(abs_path / "assets"):
        image_dir = abs_path / "assets/resource/base/image/Fish"
    else:
        image_dir = abs_path / "resource/base/image/Fish"
    valid_region_left_img = image_dir / "valid_region_left.png"
    valid_region_right_img = image_dir / "valid_region_right.png"
    slider_img = image_dir / "slider.png"
    success_catch_img = image_dir / "success_catch.png"

    slider_template = cv2.imread(str(slider_img), cv2.IMREAD_COLOR)
    valid_region_left_template = cv2.imread(
        str(valid_region_left_img), cv2.IMREAD_COLOR
    )
    valid_region_right_template = cv2.imread(
        str(valid_region_right_img), cv2.IMREAD_COLOR
    )
    success_catch_template = cv2.imread(str(success_catch_img), cv2.IMREAD_COLOR)

    def run(
        self, context: Context, _argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("钓鱼动作开始 (auto_fish_new)")
        controller = context.tasker.controller

        KEY_A = 65
        KEY_D = 68
        KEY_F = 70

        success_region = [350, 150, 580, 50]
        game_region = [395, 40, 490, 20]
        deadzone = 15

        logger.info("阶段 1/2: 抛竿后等待鱼上钩")

        # --- 等待鱼上钩 ---
        wait_frame = 0
        while True:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            time.sleep(0.001)
            img = get_image(controller)
            wait_frame += 1
            m_catch, catch_score, _, _ = match_template_in_region(
                img, success_region, self.success_catch_template, 0.7
            )
            if wait_frame > 300:
                logger.warning(f"  等待鱼上钩超时 (f={wait_frame})，结束本次钓鱼")
                break
            if m_catch:
                controller.post_key_down(KEY_F)
                time.sleep(0.1)
                controller.post_key_up(KEY_F)
                logger.info(f"  鱼已上钩！(f={wait_frame}, score={catch_score:.3f})")
                break

        logger.info("阶段 2/2: 进入控条小游戏")

        # --- 小游戏 ---
        frame = 0
        current_ad_key = None
        last_bar_width = 100
        last_target = (game_region[0] + game_region[2]) / 2
        last_x_slider = last_target
        slider_miss_count = 0

        def set_ad_key(key):
            nonlocal current_ad_key
            if current_ad_key == key:
                return
            if current_ad_key is not None:
                controller.post_key_up(current_ad_key)
            if key is not None:
                controller.post_key_down(key)
            current_ad_key = key

        while True:
            if context.tasker.stopping:
                set_ad_key(None)
                controller.post_key_up(KEY_A)
                controller.post_key_up(KEY_D)
                return CustomAction.RunResult(success=False)
            time.sleep(0.001)
            img = get_image(controller)
            frame += 1

            m_left, left_score, x_left, _ = match_template_in_region(
                img, game_region, self.valid_region_left_template, 0.7
            )
            m_right, right_score, x_right, _ = match_template_in_region(
                img, game_region, self.valid_region_right_template, 0.7
            )
            m_slider, slider_score, x_slider, _ = match_template_in_region(
                img, game_region, self.slider_template, 0.9
            )

            if frame % 10 == 0:
                if current_ad_key is not None:
                    controller.post_key_up(current_ad_key)
                controller.post_key_down(KEY_F)
                time.sleep(0.05)
                controller.post_key_up(KEY_F)
                if current_ad_key is not None:
                    controller.post_key_down(current_ad_key)

            if m_slider:
                slider_miss_count = 0
                last_x_slider = x_slider
            else:
                slider_miss_count += 1
                if slider_miss_count >= 30:
                    set_ad_key(None)
                    controller.post_key_up(KEY_F)
                    logger.info(
                        f"  滑块连续 {slider_miss_count} 帧丢失，本次钓鱼结束 (success)"
                    )
                    return CustomAction.RunResult(success=True)
                x_slider = last_x_slider

            if frame > 300:
                set_ad_key(None)
                controller.post_key_up(KEY_A)
                controller.post_key_up(KEY_D)
                controller.post_key_up(KEY_F)
                logger.warning(f"  控条阶段超时 (f={frame})，本次钓鱼结束 (failure)")
                return CustomAction.RunResult(success=False)

            if m_left and m_right:
                last_bar_width = x_right - x_left
                target = (x_left + x_right) / 2
                last_target = target
            elif m_left and not m_right:
                target = x_left + last_bar_width / 2
                last_target = target
            elif not m_left and m_right:
                target = x_right - last_bar_width / 2
                last_target = target
            else:
                target = last_target

            offset = x_slider - target
            prev_key = current_ad_key
            if offset > deadzone:
                set_ad_key(KEY_A)
            elif offset < -deadzone:
                set_ad_key(KEY_D)
            else:
                set_ad_key(None)

            if frame % 30 == 0 or current_ad_key != prev_key:
                key_name = {None: "-", KEY_A: "A", KEY_D: "D"}.get(current_ad_key, "?")
                logger.debug(
                    f"  f={frame} slider(x={x_slider:.0f} s={slider_score:.2f}) "
                    f"L({m_left} s={left_score:.2f}) R({m_right} s={right_score:.2f}) "
                    f"bar_w={last_bar_width:.0f} target={target:.0f} offset={offset:+.0f} key={key_name}"
                )
