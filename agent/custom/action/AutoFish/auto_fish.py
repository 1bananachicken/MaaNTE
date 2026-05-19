import time
import json

from ..Common.utils import get_image
from ..Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = get_logger(__name__)


@AgentServer.custom_action("auto_fish")
class AutoFish(CustomAction):

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("=== Autofish Action Started ===")
        controller = context.tasker.controller

        fishing_count = 10
        check_freq = 0.001
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                fishing_count = params.get("count", 10)
                check_freq = params.get("freq", 0.001)
            except:
                pass

        KEY_A = 65
        KEY_D = 68
        KEY_F = 70
        KEY_ESC = 27

        game_region = [401, 39, 481, 24]

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
                settle_result = context.run_recognition("SceneClickBlankToExit", img)
                if not (settle_result and settle_result.hit):
                    return True
                time.sleep(interval)

            return False

        def ensure_fish_game():
            for _ in range(10):
                img = get_image(controller)

                settle_result = context.run_recognition("SceneClickBlankToExit", img)
                if settle_result and settle_result.hit:
                    logger.debug(
                        "Found settlement screen during check, pressing ESC to close..."
                    )
                    press_esc()
                    wait_until_settlement_disappears()
                    continue

                game_result = context.run_recognition("FishGameSign3", img)
                if game_result and game_result.hit:
                    return True

                prepare_result = context.run_recognition("FishPrepareStartButton", img)
                if prepare_result and prepare_result.hit:
                    logger.debug("On FishPrepare screen, pressing start...")
                    controller.post_click(
                        prepare_result.box.x + 15, prepare_result.box.y + 15
                    )
                    time.sleep(1.5)
                    return True

                time.sleep(0.1)

            logger.error("ERROR: Not in FishGame or FishPrepare, exiting fishing.")
            return False

        for i in range(fishing_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            logger.info(f"=== Fishing {i + 1}/{fishing_count} ===")

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
                    need_bait_result = context.run_recognition("FishNeedBait", img)
                    if need_bait_result and need_bait_result.hit:
                        logger.debug("Need bait! Switching to bait handler.")
                        # 缺少鱼饵不是异常退出：这里临时改写 FishGameStart 的后续节点，
                        # 让流水线去打开鱼饵界面，优先切换万能鱼饵，必要时再购买鱼饵。
                        context.override_next("FishGameStart", ["FishHandleBaitLack"])
                        return CustomAction.RunResult(success=True)

                    time.sleep(0.1)

                logger.debug("Casting...")

                wait_start = time.time()
                m_settle_unexpected = False
                timeout_triggered = False

                while True:
                    if context.tasker.stopping:
                        return CustomAction.RunResult(success=False)

                    if time.time() - wait_start > 30:
                        logger.debug("Timeout waiting for fish to hook, recasting...")
                        timeout_triggered = True
                        break

                    time.sleep(check_freq)
                    img = get_image(controller)

                    settle_result = context.run_recognition(
                        "SceneClickBlankToExit", img
                    )
                    m_settle_unexpected = (
                        settle_result is not None and settle_result.hit
                    )
                    if m_settle_unexpected:
                        logger.debug(
                            "Unexpected settlement screen detected! Breaking to clear it."
                        )
                        break

                    catch_result = context.run_recognition("FishSuccessCatch", img)
                    if catch_result and catch_result.hit:
                        logger.debug("Fish hooked!")
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
                        settle_result = context.run_recognition(
                            "SceneClickBlankToExit", img
                        )
                        if settle_result and settle_result.hit:
                            logger.debug("Fish caught!")
                            break
                        escape_result = context.run_recognition("FishEscape", img)
                        if escape_result and escape_result.hit:
                            logger.debug("Fish escaped! Recasting...")
                            break

                    left_result = context.run_recognition("FishValidRegionLeft", img)
                    right_result = context.run_recognition("FishValidRegionRight", img)
                    slider_result = context.run_recognition("FishSlider", img)

                    m_left = left_result is not None and left_result.hit
                    m_right = right_result is not None and right_result.hit
                    m_slider = slider_result is not None and slider_result.hit

                    x_left = left_result.box.x if m_left else 0
                    x_right = right_result.box.x if m_right else 0
                    x_slider = slider_result.box.x if m_slider else 0

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
                controller.post_key_up(KEY_F)

                img = get_image(controller)
                time.sleep(0.3)
                escape_result = context.run_recognition("FishEscape", img)
                if escape_result and escape_result.hit:
                    continue
                break

            logger.debug("Finished.")

            match_settle = False
            wait_settlement_start = time.time()
            while time.time() - wait_settlement_start < 15:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                settle_result = context.run_recognition("SceneClickBlankToExit", img)
                if settle_result and settle_result.hit:
                    logger.debug("Settlement screen detected.")
                    break
                time.sleep(0.1)

            if match_settle:
                logger.debug("Closing settlement screen...")
                for _ in range(5):
                    press_esc()
                    if wait_until_settlement_disappears():
                        logger.debug("Settlement closed.")
                        break
            else:
                logger.debug("Settlement screen not detected, continuing immediately.")

        logger.info("All fishing tasks complete.")
        return CustomAction.RunResult(success=True)
