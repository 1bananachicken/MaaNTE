from __future__ import annotations

import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

try:
    from agent.custom.action.pinkpaw.pinkpaw_core3 import (
        AbortException,
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        PinkPawHeistCore3Path,
        TaskerStoppedException,
        _is_hit,
    )
except ImportError:
    from .pinkpaw_core3 import (
        AbortException,
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        PinkPawHeistCore3Path,
        TaskerStoppedException,
        _is_hit,
    )

try:
    from agent.custom.action.pinkpaw.pinkpaw_common import (
        _parse_custom_action_param,
        ensure_game_window_resolution,
    )
except ImportError:
    from .pinkpaw_common import (
        _parse_custom_action_param,
        ensure_game_window_resolution,
    )


RECOVERY_ENTRANCE_ROUTE_MINT = [
    ("w", 9.22, 0.20),
    ("d", 2.80, 0.20),
    ("w", 1.60, 0.20),
    ("d", 1.00, 0.20),
    ("w", 0.10, 0.10),
    ("w", 0.10, 0.10),
    ("d", 0.10, 0.10),
    ("d", 0.10, 0.10),
    ("s", 0.38, 0.20),
    ("a", 1.28, 0.20),
    ("w", 0.72, 0.20),
]

RECOVERY_ENTRANCE_ROUTE_MINT2 = [
    ("w", 7.42, 0.20),
    ("d", 2.80, 0.20),
    ("w", 1.60, 0.20),
    ("d", 1.00, 0.20),
    ("w", 0.10, 0.10),
    ("w", 0.10, 0.10),
    ("d", 0.10, 0.10),
    ("d", 0.10, 0.10),
    ("s", 0.38, 0.20),
    ("a", 1.08, 0.20),
    ("w", 0.52, 0.20),
]

RECOVERY_ROUTE_PROFILES = {
    "core1": {
        "label": "方案一跑图位",
        "runner": ["3"],
        "steps": RECOVERY_ENTRANCE_ROUTE_MINT,
    },
    "core2": {
        "label": "方案二跑图位",
        "runner": ["4"],
        "steps": RECOVERY_ENTRANCE_ROUTE_MINT2,
    },
    "core3": {
        "label": "方案三薄荷跑图位",
        "runner": ["3"],
        "steps": RECOVERY_ENTRANCE_ROUTE_MINT,
    },
}

FIND_XIAOZHI_RECOVERY_ATTEMPTS = 3


def _normalize_key_list(value, default):
    """把 custom_action_param 中的单个键位或键位列表统一成字符串列表。"""
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        keys = value
    else:
        keys = [value]
    result = [str(key) for key in keys if str(key)]
    return result or list(default)


class PinkPawHeistEntranceRecoveryPath(PinkPawHeistCore3Path):
    """粉爪入口恢复路线：找不到小吱时传送回大塔，并跑回小吱位置。"""

    def __init__(self, ctx: Context, params: dict | None = None):
        """读取当前粉爪方案，选择对应跑图角色和入口恢复路线。"""
        super().__init__(ctx, params=params)
        params = params or {}
        scheme = str(params.get("scheme", "core3")).lower()
        profile = RECOVERY_ROUTE_PROFILES.get(scheme, RECOVERY_ROUTE_PROFILES["core3"])
        runner = _normalize_key_list(params.get("runner"), profile["runner"])
        self.recovery_scheme = scheme
        self.recovery_route_profile = profile
        self.config[self.CONF_RUNNER] = runner

    def _has_xiaozhi_prompt(self, time_out=1.0):
        """检测当前画面是否已经能看到小吱交互提示。"""
        deadline = time.monotonic() + float(time_out)
        while time.monotonic() < deadline:
            image = self._screencap()
            if image is not None and self._recognize_once(
                "PinkPawHeist_DetectXiaoZhi", image
            ):
                return True
            self.sleep(0.1, check_reward=False, scaled=False)
        return False

    def _is_in_world_by_node(self):
        """用通用 InWorld 节点判断当前是否已经回到大世界主界面。"""
        image = self._screencap()
        if image is None:
            return False
        return self._recognize_once("InWorld", image)

    def _is_in_city_tycoon_menu(self):
        """确认当前已经打开都市大亨菜单。"""
        image = self._screencap()
        if image is None:
            return False
        return self._recognize_once("InCityTycoonMenu", image)

    def _is_in_hethereau_hobbies_menu(self):
        """确认当前已经进入都市闲趣菜单。"""
        image = self._screencap()
        if image is None:
            return False
        return self._recognize_once("InHethereauHobbiesMenu", image)

    def _is_in_heist_round(self):
        """判断当前是否仍在粉爪局内，避免把局内 UI 误当成大世界。"""
        image = self._screencap()
        if image is None:
            return False
        return self._recognize_once("PinkPawHeist_CheckReward", image)

    def _clear_recovery_menu_blockers(self):
        """清理可能挡住都市大亨入口的弹窗或残留界面。"""
        for node_name in ("SceneClickCloseButton", "SceneClickBlankToExit"):
            self.ah.run_task(
                node_name,
                pipeline_override={
                    node_name: {
                        "enabled": True,
                        "timeout": 1000,
                    }
                },
            )
            self.sleep(0.1, check_reward=False, scaled=False)

    def _close_f5_menu_with_esc(self, menu_was_confirmed=False, world_timeout=3):
        """Close the F5 menu after using it as a world-state fallback."""
        if not menu_was_confirmed:
            return False
        self.send_key(
            "esc",
            down_time=0.08,
            action_name="recovery_f5_esc",
            interval=-1,
        )
        self.sleep(0.8, check_reward=False, scaled=False)
        if self.wait_until(self._is_in_world_by_node, time_out=world_timeout):
            return True
        if not self._is_in_city_tycoon_menu():
            return True
        self._clear_recovery_menu_blockers()
        if self.wait_until(self._is_in_world_by_node, time_out=1.5):
            return True
        return not self._is_in_city_tycoon_menu()

    def _ensure_world_or_city_tycoon_menu(self, world_timeout=5, city_timeout=10):
        """确认当前可操作大世界；InWorld 识别失败时尝试打开都市大亨兜底。"""
        if self.wait_until(self._is_in_world_by_node, time_out=world_timeout):
            return True
        self.log_warning("未识别到大世界图标，尝试按 F5 打开都市大亨兜底确认")
        self._clear_recovery_menu_blockers()
        self.ah.click_key("f5")
        self.sleep(0.5, check_reward=False, scaled=False)
        menu_found = self.wait_until(
            self._is_in_city_tycoon_menu,
            time_out=city_timeout,
        )
        if not menu_found:
            self.log_warning("F5 后未确认都市大亨菜单，不按 Esc，等待后续重试")
        return self._close_f5_menu_with_esc(menu_was_confirmed=menu_found)

    def _confirm_world_after_teleport(self):
        """传送后确认回到可操作大世界；必要时用 F5 打开都市大亨再 Esc 返回。"""
        if self.wait_until(self._is_in_world_by_node, time_out=30):
            return True
        self.log_warning("传送后未识别到大世界图标，尝试 F5 兜底确认")
        for attempt in range(1, 3):
            if attempt > 1:
                self.log_warning(f"传送后大世界确认重试 {attempt}/2")
                self._clear_recovery_menu_blockers()
                if self.wait_until(self._is_in_world_by_node, time_out=3):
                    return True
            self.ah.click_key("f5")
            self.sleep(0.5, check_reward=False, scaled=False)
            menu_found = self.wait_until(self._is_in_city_tycoon_menu, time_out=5)
            if not menu_found:
                self.log_warning("F5 后未确认都市大亨菜单，不按 Esc，等待后续重试")
            if self._close_f5_menu_with_esc(
                menu_was_confirmed=menu_found,
                world_timeout=3,
            ):
                return True

        return self.wait_until(self._is_in_world_by_node, time_out=5)

    def _enter_recovery_hethereau_hobbies_menu(self):
        """稳定进入都市闲趣，避免公共跳转节点停在都市大亨菜单就继续执行。"""
        for attempt in range(1, 4):
            if self._is_in_hethereau_hobbies_menu():
                return True

            self.log_round_info(f"进入都市闲趣（第 {attempt} 次）")
            if not self._is_in_city_tycoon_menu():
                self.ah.run_task("SceneAnyEnterWorld")
                self.sleep(0.5, check_reward=False, scaled=False)
                self.ah.run_task(
                    "SceneAnyEnterCityTycoonsMenu",
                    pipeline_override={
                        "SceneAnyEnterCityTycoonsMenu": {"timeout": 15000}
                    },
                )

            if not self.wait_until(self._is_in_city_tycoon_menu, time_out=5):
                self.log_warning("未确认进入都市大亨菜单，清理界面后重试")
                self._clear_recovery_menu_blockers()
                continue

            found = self.wait_until(
                lambda: self._recognize_once(
                    "PinkPawHeist_CityTycoonToHethereauHobbies"
                ),
                time_out=5,
            )
            if not found:
                self.log_warning("未识别到都市闲趣入口，使用固定位置点击兜底")
            for click_attempt in range(1, 4):
                self.ah.click(665, 318)
                self.sleep(0.5, check_reward=False, scaled=False)
                if self.wait_until(self._is_in_hethereau_hobbies_menu, time_out=3):
                    return True
                if click_attempt < 3 and self._is_in_city_tycoon_menu():
                    self.log_warning("点击都市闲趣后未进入，当前界面再点一次")
                    continue
                break

            self.log_warning("未确认进入都市闲趣，返回大世界后重试")
            self._clear_recovery_menu_blockers()
            self.ah.run_task("SceneAnyEnterWorld")
            self.sleep(0.5, check_reward=False, scaled=False)
        return False

    def _open_recovery_heist_map_from_hobbies(self):
        """进入都市闲趣并点击粉爪大劫案，让游戏定位到粉爪传送点。"""
        self.log_round_info("打开都市闲趣并定位粉爪大劫案")
        if not self._enter_recovery_hethereau_hobbies_menu():
            raise AbortException("进入都市闲趣失败，无法恢复到小吱")
        for attempt in range(1, 3):
            self.sleep(0.3, check_reward=False, scaled=False)
            found = self.wait_until(
                lambda: self._recognize_once("PinkPawHeist_HethereauHobbiesToHeistMap"),
                time_out=8,
            )
            if not found:
                self.log_warning(
                    f"都市闲趣中未识别到粉爪大劫案，第 {attempt} 次，使用固定位置点击兜底"
                )
            if self._is_in_hethereau_hobbies_menu():
                for click_attempt in range(1, 4):
                    self.ah.click(637, 486)
                    if self.wait_until(
                        lambda: not self._is_in_hethereau_hobbies_menu(),
                        time_out=3,
                    ):
                        return True
                    if click_attempt < 3 and self._is_in_hethereau_hobbies_menu():
                        self.log_warning("点击粉爪大劫案后未进入地图，当前界面再点一次")
                        continue
                    break
            else:
                if not self._enter_recovery_hethereau_hobbies_menu():
                    break
        raise AbortException("都市闲趣中未找到粉爪大劫案")

    def _confirm_recovery_teleport(self):
        """点击地图右下角传送按钮，并等待加载回大世界。"""
        for _ in range(3):
            result = self.ah.run_task(
                "RealTimeConfirmTeleportPhone",
                pipeline_override={
                    "RealTimeConfirmTeleportPhone": {
                        "enabled": True,
                        "timeout": 2000,
                    }
                },
            )
            if _is_hit(result):
                self.sleep(1.0, check_reward=False, scaled=False)
                ret = self._confirm_world_after_teleport()
                if not ret:
                    raise AbortException("传送后未确认回到大世界")
                self.wait_team_ui_settle()
                self.sleep(1.0, check_reward=False, scaled=False)
                return ret
            self.sleep(0.5, check_reward=False, scaled=False)
        raise AbortException("传送确认失败，无法恢复到小吱")

    def _recovery_press_key(self, key, down_time, after_sleep=0):
        """恢复路线专用按键：不做副本收益检测，避免大世界寻路被中断。"""
        self.send_key_down(key)
        self.sleep(down_time, check_reward=False, scaled=False)
        self.send_key_up(key)
        if after_sleep:
            self.sleep(after_sleep, check_reward=False, scaled=False)

    def _recovery_middle_click(self, x, y, down_time=0.1, after_sleep=0):
        """恢复路线专用中键点击，用于按 OK-NTE 路线校正镜头。"""
        px = int(round(float(x) * DEFAULT_WIDTH))
        py = int(round(float(y) * DEFAULT_HEIGHT))
        self.ah.move_to(px, py)
        self.ah.mouse_down(key="middle")
        self.sleep(down_time, check_reward=False, scaled=False)
        self.ah.mouse_up(key="middle")
        if after_sleep:
            self.sleep(after_sleep, check_reward=False, scaled=False)

    def _focus_game_window(self, after_sleep=0.08):
        """Focus the game before recovery sends direct keyboard input."""
        if self.ah.focus_window():
            self.sleep(after_sleep, check_reward=False, scaled=False)
            return True
        return False

    def _leave_unexpected_heist_round(self):
        """恢复入口发现仍在局内时，先退出本局，避免在局内误开大世界恢复。"""
        if not self._is_in_heist_round():
            return False
        self.log_round_info("检测到仍在粉爪局内，先退出本局")
        self._focus_game_window()
        self._release_held_keys()
        self.ah.release_controls()
        for _ in range(4):
            self.send_key("esc")
            self.sleep(1.0, check_reward=False, scaled=False)
        self.ah.run_task("PinkPawHeist_Once")
        self.sleep(5.0, check_reward=False, scaled=False)
        self.ah.run_task("SceneAnyEnterWorld")
        return True

    def _leave_unexpected_heist_round_if_needed(self):
        """如仍在粉爪局内则退出，并确认回到可操作的大世界或都市大亨菜单。"""
        if not self._leave_unexpected_heist_round():
            return False
        if not self.wait_until(
            lambda: not self._is_in_heist_round(),
            time_out=10,
            settle_time=1.0,
        ):
            raise AbortException("仍在粉爪局内，暂不执行传送恢复")
        if not self._ensure_world_or_city_tycoon_menu(
            world_timeout=10, city_timeout=10
        ):
            raise AbortException("退出粉爪局内后未确认大世界或都市大亨菜单")
        return True

    def _run_heist_entrance_path_from_teleport(self):
        """从粉爪传送点跑回小吱位置。"""
        runner = "/".join(self.config.get(self.CONF_RUNNER, []))
        self.log_round_info(
            f"正在寻路到小吱（{self.recovery_route_profile['label']}：{runner}号位）"
        )
        self.sleep(0.50, check_reward=False, scaled=False)
        self.switch_to_runner()
        self.sleep(0.20, check_reward=False, scaled=False)
        for key, down_time, after_sleep in self.recovery_route_profile["steps"]:
            self._recovery_press_key(key, down_time, after_sleep)
        self.log_round_info("完成寻路到小吱")
        self.sleep(0.30, check_reward=False, scaled=False)
        return True

    def recover_to_heist_entrance(self):
        """找不到小吱时，传送回粉爪大塔并跑到小吱。"""
        self.log_round_info("找不到小吱，尝试恢复到粉爪入口")
        self._release_held_keys()
        self.ah.release_controls()
        self.ah.run_task("SceneAnyEnterWorld")
        if (
            not self._leave_unexpected_heist_round_if_needed()
            and not self._ensure_world_or_city_tycoon_menu(
                world_timeout=5, city_timeout=10
            )
        ):
            self.ah.run_task("SceneAnyEnterWorld")
            if not self._ensure_world_or_city_tycoon_menu(
                world_timeout=10, city_timeout=10
            ):
                raise AbortException("仍未确认大世界或都市大亨菜单，暂不执行传送恢复")
        self._leave_unexpected_heist_round_if_needed()
        if not self._is_in_city_tycoon_menu() and self._has_xiaozhi_prompt(
            time_out=5.0
        ):
            self.log_round_info("清理界面后已找到小吱")
            return True
        self._open_recovery_heist_map_from_hobbies()
        self._confirm_recovery_teleport()
        self._run_heist_entrance_path_from_teleport()
        self._release_held_keys()
        self.ah.release_controls()
        return True


@AgentServer.custom_action("PinkPawHeistReturnToEntranceAction")
class PinkPawHeistReturnToEntranceAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        """找不到小吱时的恢复入口：传送到粉爪大塔并跑回小吱位置。"""
        params = _parse_custom_action_param(argv, log_prefix="[PinkPawHeist/Recovery]")
        path = PinkPawHeistEntranceRecoveryPath(context, params=params)
        try:
            if path.auto_resize_game_window and ensure_game_window_resolution:
                ensure_game_window_resolution(DEFAULT_WIDTH, DEFAULT_HEIGHT)
            path.recover_to_heist_entrance()
            return CustomAction.RunResult(success=True)
        except TaskerStoppedException as exc:
            print(f"[PinkPawHeist/Recovery] return to entrance stopped: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)
        except Exception as exc:
            print(f"[PinkPawHeist/Recovery] return to entrance failed: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)


@AgentServer.custom_action("PinkPawHeistFindXiaoZhiAction")
class PinkPawHeistFindXiaoZhiAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        """寻找小吱入口；找不到时先恢复到粉爪入口，再重新确认交互提示。"""
        params = _parse_custom_action_param(argv, log_prefix="[PinkPawHeist/Recovery]")
        path = PinkPawHeistEntranceRecoveryPath(context, params=params)
        try:
            if path.auto_resize_game_window and ensure_game_window_resolution:
                ensure_game_window_resolution(DEFAULT_WIDTH, DEFAULT_HEIGHT)
            path.log_round_info("开始寻找小吱")

            # 先尝试识别"我要参加"（仅OCR检查，不执行节点动作）
            if path._recognize_once("PinkPawHeist_OCR_Join"):
                path.log_round_info(
                    "未找到小吱，但识别到'我要参加'，视为已完成寻找小吱并按F"
                )
                return CustomAction.RunResult(success=True)
            path.ah.run_task("SceneAnyEnterWorld")
            if path._has_xiaozhi_prompt(time_out=10.0):
                path.log_round_info("成功找到小吱，开始任务")
                return CustomAction.RunResult(success=True)

            # 先尝试识别"我要参加"（仅OCR检查，不执行节点动作）
            if path._recognize_once("PinkPawHeist_OCR_Join"):
                path.log_round_info(
                    "未找到小吱，但识别到'我要参加'，视为已完成寻找小吱并按F"
                )
                return CustomAction.RunResult(success=True)

            for attempt in range(1, FIND_XIAOZHI_RECOVERY_ATTEMPTS + 1):
                path.log_round_info(
                    f"未找到小吱，尝试恢复到粉爪入口（第 {attempt}/{FIND_XIAOZHI_RECOVERY_ATTEMPTS} 次）"
                )
                recovered = False
                try:
                    path.recover_to_heist_entrance()
                    recovered = True
                except TaskerStoppedException:
                    raise
                except Exception as exc:
                    path.log_warning(f"第 {attempt} 次恢复失败：{exc}")
                    print(
                        f"[PinkPawHeist/Recovery] recover attempt {attempt} failed: {exc}"
                    )
                finally:
                    path._release_held_keys()
                    path.ah.release_controls()

                try:
                    path.ah.run_task("SceneAnyEnterWorld")
                    path.sleep(0.5, check_reward=False, scaled=False)
                    path._clear_recovery_menu_blockers()
                except TaskerStoppedException:
                    raise
                except Exception as exc:
                    path.log_warning(f"第 {attempt} 次恢复后清理界面失败：{exc}")

                if path._has_xiaozhi_prompt(time_out=10.0):
                    path.log_round_info("恢复后已找到小吱，开始任务")
                    return CustomAction.RunResult(success=True)

                if recovered:
                    path.log_warning(f"第 {attempt} 次恢复已执行，但仍未找到小吱")
                else:
                    path.log_warning(f"第 {attempt} 次恢复失败后仍未找到小吱")

            path.log_warning(
                f"连续 {FIND_XIAOZHI_RECOVERY_ATTEMPTS} 次恢复后仍未找到小吱"
            )
            return CustomAction.RunResult(success=False)
        except TaskerStoppedException as exc:
            print(f"[PinkPawHeist/Recovery] find XiaoZhi stopped: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)
        except Exception as exc:
            print(f"[PinkPawHeist/Recovery] find XiaoZhi failed: {exc}")
            path._release_held_keys()
            path.ah.release_controls()
            return CustomAction.RunResult(success=False)
