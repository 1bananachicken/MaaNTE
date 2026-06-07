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
        _as_bgr_image,
        _crop_roi,
        _is_hit,
        _load_fast_template,
        _scale_roi,
        cv2,
        Image,
        np,
    )
except ImportError:
    from .pinkpaw_core3 import (
        AbortException,
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        PinkPawHeistCore3Path,
        TaskerStoppedException,
        _as_bgr_image,
        _crop_roi,
        _is_hit,
        _load_fast_template,
        _scale_roi,
        cv2,
        Image,
        np,
    )


RECOVERY_MAP_OPEN_ROI = [0, 560, 240, 160]
RECOVERY_MAP_TELEPORT_ROI = [0, 60, 1280, 600]
RECOVERY_MAP_OPEN_TEMPLATES = ["map_city_tycoon_activities.png"]
RECOVERY_MAP_TELEPORT_TEMPLATES = ["map_big_teleport.png"]
RECOVERY_MAP_OPEN_THRESHOLD = 0.72
RECOVERY_MAP_TELEPORT_THRESHOLD = 0.62
RECOVERY_MAP_TELEPORT_SCALES = (0.65, 0.78, 0.9, 1.0, 1.12, 1.28, 1.45)


class PinkPawHeistEntranceRecoveryPath(PinkPawHeistCore3Path):
    """粉爪入口恢复路线：找不到小吱时传送回大塔，并跑回小吱位置。"""

    def _find_template_point(
        self,
        image,
        template_names,
        roi=None,
        threshold=0.8,
        nearest_to=None,
        scales=(1.0,),
        debug_name=None,
        prefer_score=False,
    ):
        """在指定截图和 ROI 内匹配模板，并返回命中的屏幕中心点。"""
        if cv2 is None or Image is None or np is None:
            return None
        bgr = _as_bgr_image(image)
        if bgr is None:
            return None
        ih, iw = bgr.shape[:2]
        if roi is None:
            scaled_roi = [0, 0, iw, ih]
        else:
            scaled_roi = _scale_roi(roi, bgr)
        crop = _crop_roi(bgr, scaled_roi)
        if crop is None:
            return None

        scene_gray = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2GRAY)
        sx = iw / DEFAULT_WIDTH
        sy = ih / DEFAULT_HEIGHT
        if nearest_to is None:
            nearest_to = (iw / 2.0, ih / 2.0)

        best = None
        best_score = None
        best_score_name = None
        for name in template_names:
            template = _load_fast_template(name)
            if template is None:
                continue
            base_templ = template["cv_bgr"]
            base_w = max(1, int(round(base_templ.shape[1] * sx)))
            base_h = max(1, int(round(base_templ.shape[0] * sy)))
            for scale in scales:
                tw = max(1, int(round(base_w * float(scale))))
                th = max(1, int(round(base_h * float(scale))))
                if scene_gray.shape[0] < th or scene_gray.shape[1] < tw:
                    continue
                if (tw, th) != (base_templ.shape[1], base_templ.shape[0]):
                    templ = cv2.resize(
                        base_templ, (tw, th), interpolation=cv2.INTER_AREA
                    )
                else:
                    templ = base_templ

                templ_gray = cv2.cvtColor(
                    np.ascontiguousarray(templ), cv2.COLOR_BGR2GRAY
                )
                scores = cv2.matchTemplate(scene_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(scores)
                if best_score is None or max_val > best_score:
                    best_score = float(max_val)
                    best_score_name = f"{name}@{scale:.2f}"
                ys, xs = np.where(scores >= threshold)
                for y, x in zip(ys, xs):
                    score = float(scores[y, x])
                    cx = float(scaled_roi[0] + x + tw / 2.0)
                    cy = float(scaled_roi[1] + y + th / 2.0)
                    dist = (cx - nearest_to[0]) ** 2 + (cy - nearest_to[1]) ** 2
                    if prefer_score:
                        candidate = (
                            -score,
                            dist,
                            int(round(cx)),
                            int(round(cy)),
                            f"{name}@{scale:.2f}",
                        )
                    else:
                        candidate = (
                            dist,
                            -score,
                            int(round(cx)),
                            int(round(cy)),
                            f"{name}@{scale:.2f}",
                        )
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            if debug_name and best_score is not None:
                self.log_warning(
                    f"{debug_name} template not found, best {best_score:.3f} {best_score_name}"
                )
            return None
        if debug_name:
            score = -best[0] if prefer_score else -best[1]
            self.log_info(
                f"{debug_name} selected {best[4]} score {score:.3f} at {(best[2], best[3])}"
            )
        return best[2], best[3]

    def _is_recovery_map_open(self):
        """用地图左下角城建图标模板判断大地图是否已经打开。"""
        image = self._screencap()
        return (
            self._find_template_point(
                image,
                RECOVERY_MAP_OPEN_TEMPLATES,
                roi=RECOVERY_MAP_OPEN_ROI,
                threshold=RECOVERY_MAP_OPEN_THRESHOLD,
            )
            is not None
        )

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

    def _open_recovery_world_map(self):
        """回到大世界后按 M 打开地图，并等待地图 UI 稳定出现。"""
        self.ah.run_task("SceneAnyEnterWorld")
        self.sleep(0.5, check_reward=False, scaled=False)
        if not self._is_in_world_by_node():
            raise AbortException("当前不在大世界，无法恢复到小吱")

        for _ in range(3):
            self.send_key("m")
            if self.wait_until(
                self._is_recovery_map_open,
                time_out=3,
                raise_if_not_found=False,
            ):
                self.sleep(0.8, check_reward=False, scaled=False)
                return True
            self.sleep(0.3, check_reward=False, scaled=False)
        raise AbortException("打开地图失败，无法恢复到小吱")

    def _click_nearest_recovery_teleport(self):
        """在地图中查找大塔传送点图标并点击。"""
        point = None
        for _ in range(3):
            image = self._screencap()
            point = self._find_template_point(
                image,
                RECOVERY_MAP_TELEPORT_TEMPLATES,
                roi=RECOVERY_MAP_TELEPORT_ROI,
                threshold=RECOVERY_MAP_TELEPORT_THRESHOLD,
                scales=RECOVERY_MAP_TELEPORT_SCALES,
                debug_name="recovery teleport",
                prefer_score=True,
            )
            if point is not None:
                break
            self.sleep(0.5, check_reward=False, scaled=False)
        if point is None:
            raise AbortException("地图中未找到可用传送点")
        self.log_round_info(f"点击大塔传送点 {point}")
        self.ah.click(point[0], point[1])
        self.sleep(0.5, check_reward=False, scaled=False)
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
        """从粉爪大塔传送点按 OK-NTE 入口路径跑回小吱位置。"""
        self.log_round_info("正在寻路到小吱")
        self.sleep(0.50, check_reward=False, scaled=False)
        self.switch_to_runner()
        self.sleep(0.20, check_reward=False, scaled=False)
        self._recovery_press_key("s", 0.73, 0.20)
        self._recovery_press_key("a", 0.10, 0.10)
        self._recovery_press_key("a", 0.10, 0.10)
        self._recovery_press_key("a", 2.40, 0.20)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("w", 5.20, 0.20)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("d", 2.50, 0.20)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("w", 0.10, 0.10)
        self._recovery_press_key("w", 10.50, 0.20)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_middle_click(0.600, 0.001, down_time=0.10, after_sleep=0.10)
        self._recovery_press_key("w", 12.86, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_press_key("d", 0.10, 0.10)
        self._recovery_middle_click(0.600, 0.001, down_time=0.10, after_sleep=0.10)
        self._recovery_press_key("w", 9.32, 0.20)
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
        self._open_recovery_world_map()
        self._click_nearest_recovery_teleport()
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
