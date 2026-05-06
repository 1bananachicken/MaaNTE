import cv2
import time
import json

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region
from ..Common.logger import get_logger
from ..Common.progress import ProgressReporter

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = get_logger("auto_fish")
progress = ProgressReporter("钓鱼", logger)


@AgentServer.custom_action("auto_fish")
class AutoFish(CustomAction):
    abs_path = Path(__file__).parents[4]
    if Path.exists(abs_path / "assets"):
        image_dir = abs_path / "assets/resource/base/image/Fish"
    else:
        image_dir = abs_path / "resource/base/image/Fish"
    settlement_img = image_dir / "settlement_blank.png"
    valid_region_left_img = image_dir / "valid_region_left.png"
    valid_region_right_img = image_dir / "valid_region_right.png"
    slider_img = image_dir / "slider.png"
    success_catch_img = image_dir / "success_catch.png"
    escape_img = image_dir / "escape.png"
    prepare_start_img = image_dir / "FishPrepareStartButton.png"
    fish_game_sign_img = image_dir / "FishGameSign3.png"
    need_bait_img = image_dir / "need_bait.png"

    slider_template = cv2.imread(str(slider_img), cv2.IMREAD_COLOR)
    valid_region_left_template = cv2.imread(
        str(valid_region_left_img), cv2.IMREAD_COLOR
    )
    valid_region_right_template = cv2.imread(
        str(valid_region_right_img), cv2.IMREAD_COLOR
    )
    settlement_template = cv2.imread(str(settlement_img), cv2.IMREAD_COLOR)
    success_catch_template = cv2.imread(str(success_catch_img), cv2.IMREAD_COLOR)
    escape_template = cv2.imread(str(escape_img), cv2.IMREAD_COLOR)
    prepare_start_template = cv2.imread(str(prepare_start_img), cv2.IMREAD_COLOR)
    fish_game_sign_template = cv2.imread(str(fish_game_sign_img), cv2.IMREAD_COLOR)
    need_bait_template = cv2.imread(str(need_bait_img), cv2.IMREAD_COLOR)

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("=== Autofish Action Started ===")
        controller = context.tasker.controller

        fishing_count = 10
        loop_count = 0
        check_freq = 0.001
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                fishing_count = params.get("count", 10)
                loop_count = params.get("loop_count", 0)
                check_freq = params.get("freq", 0.001)
            except (json.JSONDecodeError, TypeError):
                pass

        KEY_A = 65
        KEY_D = 68
        KEY_F = 70
        KEY_ESC = 27

        success_region = [520, 160, 265, 30]
        settlement_region = [566, 642, 150, 23]
        game_region = [401, 39, 481, 24]
        escape_region = [590, 349, 99, 22]
        prepare_region = [908, 602, 339, 52]
        fish_game_sign_region = [1141, 609, 87, 84]
        fish_game_sign_region_2 = [1224, 27, 30, 30]
        need_bait_region = [610, 350, 141, 21]

        def press_esc():
            controller.post_key_down(KEY_ESC)
            time.sleep(0.1)
            controller.post_key_up(KEY_ESC)

        def wait_until_settlement_disappears(timeout=1.0, interval=0.05):
            wait_start = time.time()
            while time.time() - wait_start < timeout:
                if context.tasker.stopping:
                    return False

                img = get_image(controller)
                matched, _, _, _ = match_template_in_region(
                    img, settlement_region, self.settlement_template, 0.8
                )

                if not matched:
                    return True
                time.sleep(interval)

            return False

        def ensure_fish_game():
            for _ in range(10):
                img = get_image(controller)

                m_settle, _, _, _ = match_template_in_region(
                    img, settlement_region, self.settlement_template, 0.8
                )
                if m_settle:
                    logger.info(
                        "  Found settlement screen during check, pressing ESC to close..."
                    )
                    press_esc()
                    wait_until_settlement_disappears()
                    continue

                m_game, game_prob, _, _ = match_template_in_region(
                    img,
                    fish_game_sign_region_2,
                    self.fish_game_sign_template,
                    0.6,
                    green_mask=True,
                )
                logger.debug(f"  Checking for FishGame screen, probability: {game_prob:.2f}")
                if m_game:
                    return True

                m_prepare, _, x, y = match_template_in_region(
                    img, prepare_region, self.prepare_start_template, 0.7
                )
                if m_prepare:
                    logger.info("  On FishPrepare screen, pressing start...")
                    controller.post_click(x + 15, y + 15)
                    time.sleep(1.5)
                    return True

                time.sleep(0.1)

            logger.info("  ERROR: Not in FishGame or FishPrepare, exiting fishing.")
            return False

        for i in range(fishing_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            if loop_count > 0:
                progress.report_combined(context, "FishStart", loop_count, i + 1, fishing_count)
            else:
                progress.report_count(i + 1, fishing_count)

            if not ensure_fish_game():
                return CustomAction.RunResult(success=False)

            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                for _ in range(5):
                    controller.post_key_down(KEY_F)
                    time.sleep(0.1)
                    controller.post_key_up(KEY_F)

                for _ in range(5):
                    img = get_image(controller)
                    m_need_bait, prob, _, _ = match_template_in_region(
                        img, need_bait_region, self.need_bait_template, 0.7
                    )
                    logger.debug(f"  Checking for bait, probability: {prob:.2f}")
                    if m_need_bait:
                        logger.info("  Need bait! Stopping fishing.")
                        return CustomAction.RunResult(success=False)

                    time.sleep(0.1)

                logger.info("  Casting...")

                wait_start = time.time()
                m_settle_unexpected = False
                timeout_triggered = False

                while True:
                    if context.tasker.stopping:
                        return CustomAction.RunResult(success=False)

                    if time.time() - wait_start > 30:
                        logger.info("  Timeout waiting for fish to hook, recasting...")
                        timeout_triggered = True
                        break

                    time.sleep(check_freq)
                    img = get_image(controller)

                    m_settle_unexpected, _, _, _ = match_template_in_region(
                        img, settlement_region, self.settlement_template, 0.8
                    )
                    if m_settle_unexpected:
                        logger.info(
                            "  Unexpected settlement screen detected! Breaking to clear it."
                        )
                        break

                    m_catch, _, _, _ = match_template_in_region(
                        img, success_region, self.success_catch_template, 0.7
                    )
                    if m_catch:
                        logger.info("  Fish hooked!")
                        break

                if m_settle_unexpected or timeout_triggered:
                    if m_settle_unexpected:
                        press_esc()
                        wait_until_settlement_disappears()
                    continue

                start_time = time.time()
                frame = 0
                deadzone = 15
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

                while time.time() - start_time < 100:
                    if context.tasker.stopping:
                        set_ad_key(None)
                        return CustomAction.RunResult(success=False)
                    time.sleep(check_freq)
                    img = get_image(controller)
                    frame += 1

                    if frame % 10 == 0:
                        m_settle, _, _, _ = match_template_in_region(
                            img, settlement_region, self.settlement_template, 0.8
                        )
                        if m_settle:
                            logger.info("  Fish caught!")
                            break
                        m_escape, _, _, _ = match_template_in_region(
                            img, escape_region, self.escape_template, 0.8
                        )
                        if m_escape:
                            logger.info("  Fish escaped! Recasting...")
                            break

                    m_left, _, x_left, _ = match_template_in_region(
                        img, game_region, self.valid_region_left_template, 0.7
                    )
                    m_right, _, x_right, _ = match_template_in_region(
                        img, game_region, self.valid_region_right_template, 0.7
                    )
                    m_slider, _, x_slider, _ = match_template_in_region(
                        img, game_region, self.slider_template, 0.7
                    )

                    if frame % 10 == 0:
                        if current_ad_key is not None:
                            controller.post_key_up(current_ad_key)

                        if current_ad_key is not None:
                            controller.post_key_down(current_ad_key)

                    if m_slider:
                        slider_miss_count = 0
                        last_x_slider = x_slider
                    else:
                        slider_miss_count += 1
                        if slider_miss_count < 15:
                            x_slider = last_x_slider
                        else:
                            x_slider = None

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

                    if target is not None and x_slider is not None:
                        offset = x_slider - target
                        if offset > deadzone:
                            set_ad_key(KEY_A)
                        elif offset < -deadzone:
                            set_ad_key(KEY_D)
                        else:
                            set_ad_key(None)
                    else:
                        set_ad_key(None)

                set_ad_key(None)

                img = get_image(controller)
                time.sleep(0.3)
                m_escape, _, _, _ = match_template_in_region(
                    img, escape_region, self.escape_template, 0.8
                )
                if m_escape:
                    continue
                break

            logger.info("  Finished.")

            match_settle = False
            wait_settlement_start = time.time()
            while time.time() - wait_settlement_start < 5:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                match_settle, settle_prob, _, _ = match_template_in_region(
                    img, settlement_region, self.settlement_template, 0.8
                )
                logger.debug(
                    f"  Checking for settlement screen, probability: {settle_prob:.2f}"
                )
                if match_settle:
                    logger.info("  Settlement screen detected.")
                    break
                time.sleep(0.05)

            if match_settle:
                logger.info("  Closing settlement screen...")
                for _ in range(5):
                    press_esc()
                    if wait_until_settlement_disappears():
                        logger.info("  Settlement closed.")
                        break
            else:
                logger.info("  Settlement screen not detected, continuing immediately.")

        logger.info("All fishing tasks complete.")
        return CustomAction.RunResult(success=True)
