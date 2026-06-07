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


class PinkPawHeistEntranceRecoveryPath(PinkPawHeistCore3Path):
    """粉爪入口恢复路线：找不到小吱时传送回大塔，并跑回小吱位置。"""

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

    def _open_recovery_heist_map_from_hobbies(self):
        """进入都市闲趣并点击粉爪大劫案，让游戏定位到粉爪传送点。"""
        self.log_round_info("打开都市闲趣并定位粉爪大劫案")
        if not _is_hit(self.ah.run_task("SceneAnyEnterHethereauHobbiesMenu")):
            raise AbortException("进入都市闲趣失败，无法恢复到小吱")
        self.sleep(0.3, check_reward=False, scaled=False)
        result = self.ah.run_task(
            "PinkPawHeist_HethereauHobbiesToHeistMap",
            pipeline_override={
                "PinkPawHeist_HethereauHobbiesToHeistMap": {
                    "enabled": True,
                    "timeout": 5000,
                }
            },
        )
        if not _is_hit(result):
            raise AbortException("都市闲趣中未找到粉爪大劫案")
        self.sleep(1.0, check_reward=False, scaled=False)
        return True

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
                ret = self.wait_until(
                    self._is_in_world_by_node,
                    time_out=30,
                    raise_if_not_found=True,
                )
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

    def _run_heist_entrance_path_from_teleport(self):
        """从粉爪传送点跑回小吱位置。"""
        self.log_round_info("正在寻路到小吱")
        self.sleep(0.50, check_reward=False, scaled=False)
        self.switch_to_runner()
        self.sleep(0.20, check_reward=False, scaled=False)
        self._recovery_press_key("w", 9.22, 0.20)
        self._recovery_press_key("d", 2.80, 0.20)
        self._recovery_press_key("w", 1.60, 0.20)
        self._recovery_press_key("d", 1.00, 0.20)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("s", 0.38, 0.20)
        self._recovery_press_key("a", 1.28, 0.20)
        self._recovery_press_key("w", 0.72, 0.20)
        self.log_round_info("完成寻路到小吱")
        self.sleep(0.30, check_reward=False, scaled=False)
        return True

    def recover_to_heist_entrance(self):
        """找不到小吱时，传送回粉爪大塔并跑到小吱。"""
        self.log_round_info("找不到小吱，尝试恢复到粉爪入口")
        self._release_held_keys()
        self.ah.release_controls()
        self.ah.run_task("SceneAnyEnterWorld")
        self.sleep(0.5, check_reward=False, scaled=False)
        if self._has_xiaozhi_prompt(time_out=5.0):
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
        path = PinkPawHeistEntranceRecoveryPath(context, params={})
        try:
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
        path = PinkPawHeistEntranceRecoveryPath(context, params={})
        try:
            path.log_round_info("开始寻找小吱")
            if path._has_xiaozhi_prompt(time_out=30.0):
                path.log_round_info("成功找到小吱，开始任务")
                return CustomAction.RunResult(success=True)

            path.log_round_info("未找到小吱，尝试恢复到粉爪入口")
            path.recover_to_heist_entrance()
            if path._has_xiaozhi_prompt(time_out=10.0):
                path.log_round_info("恢复后已找到小吱，开始任务")
                return CustomAction.RunResult(success=True)

            path.log_warning("恢复后仍未找到小吱")
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
