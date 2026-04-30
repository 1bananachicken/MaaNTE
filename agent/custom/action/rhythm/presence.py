"""
游戏场景状态检测：识别 OTHER / SONG_SELECT / PLAYING / RESULTS 四种状态。

原理：
1. 子模块模板匹配（可选）：将 assets/scene_templates/<kind>/ 下的每张图片视为
    一个独立子模块（如按钮、图标、标签），用 cv2.matchTemplate 在画面中逐一匹配。
    至少 match_vote_min 个子模块命中则判定为对应场景。需 scene.template_match_enabled=true。
2. Laplacian 方差检测：在四条轨道对应的画面底部各取一块正方形灰度图，
    计算 Laplacian 方差，作为 PLAYING 的兜底识别。

状态分类逻辑（优先级从高到低）：
  song_select 子模块匹配成功            → SONG_SELECT
  results 子模块匹配成功                 → RESULTS
  playing 子模块匹配成功                 → PLAYING
  四鼓位 Laplacian 方差全部达标          → PLAYING
  以上都不满足                           → OTHER
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .assets import list_scene_templates
from .lanes import LaneLayout

logger = logging.getLogger(__name__)

STATE_OTHER = "other"
STATE_SONG_SELECT = "song_select"
STATE_PLAYING = "playing"
STATE_RESULTS = "results"


class SceneGate:
    """带滞后的场景状态分类器。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        p = cfg.get("presence") or {}
        self.enabled = bool(p.get("enabled", True))
        self.drum_y_frac = float(p.get("drum_center_y_frac", 0.88))
        self.patch_half_w_frac = float(p.get("patch_half_width_frac", 0.032))
        self.min_laplace_var = float(p.get("min_laplace_variance", 95.0))
        self.min_mean_gray = float(p.get("min_mean_gray", 18.0))
        self.max_mean_gray = float(p.get("max_mean_gray", 255.0))
        self.arm_after_good_frames = max(1, int(p.get("arm_after_good_frames", 1)))
        self.disarm_after_bad_frames = max(1, int(p.get("disarm_after_bad_frames", 5)))

        sc = cfg.get("scene") or {}
        self._template_match_enabled = bool(sc.get("template_match_enabled", False))
        self._song_select_thresh = float(sc.get("song_select_match_threshold", 0.75))
        self._results_thresh = float(sc.get("results_match_threshold", 0.75))
        self._playing_thresh = float(sc.get("playing_match_threshold", 0.75))
        self._state_confirm_frames = max(1, int(sc.get("state_confirm_frames", 2)))
        self._match_blur_ksize = int(sc.get("match_blur_ksize", 3))
        self._match_downscale = float(sc.get("match_downscale", 1.0))
        self._match_vote_min = max(1, int(sc.get("match_vote_min", 2)))
        self._song_select_roi = sc.get("song_select_roi")
        self._results_roi = sc.get("results_roi")
        self._playing_roi = sc.get("playing_roi")
        self._match_skip_playing = bool(sc.get("match_skip_playing", True))

        if self._template_match_enabled:
            self._song_select_tpls = self._load_and_prepare_templates("song_select")
            self._results_tpls = self._load_and_prepare_templates("results")
            self._playing_tpls = self._load_and_prepare_templates("playing")
        else:
            self._song_select_tpls = []
            self._results_tpls = []
            self._playing_tpls = []

        self._state: str = STATE_OTHER
        self._target_state: str = STATE_OTHER
        self._state_streak: int = 0
        self._armed = False
        self._good_streak = 0
        self._bad_streak = 0

    @property
    def state(self) -> str:
        return self._state

    def step(
        self,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return STATE_PLAYING, {
                "enabled": False,
                "state": STATE_PLAYING,
                "armed": True,
                "transitioned": False,
                "state_transitioned": False,
                "per_ok": [True, True, True, True],
                "lap_vars": [0.0, 0.0, 0.0, 0.0],
                "mean_grays": [0.0, 0.0, 0.0, 0.0],
                "song_select_tpl_val": 0.0,
                "results_tpl_val": 0.0,
                "playing_tpl_val": 0.0,
            }

        per_ok, lap_vars, mean_grays = self._measure_patches(frame_bgr, layout)
        drums_present = all(per_ok)

        in_playing = self._armed and self._state == STATE_PLAYING
        match_blocked = (
            not self._template_match_enabled
            or (in_playing and self._match_skip_playing)
        )

        if not match_blocked:
            ss_ok, ss_val, ss_name = self._match_templates(
                frame_bgr,
                self._song_select_tpls,
                self._song_select_thresh,
                roi_frac=self._song_select_roi,
            )
            rs_ok, rs_val, rs_name = self._match_templates(
                frame_bgr,
                self._results_tpls,
                self._results_thresh,
                roi_frac=self._results_roi,
            )
            pl_ok, pl_val, pl_name = self._match_templates(
                frame_bgr,
                self._playing_tpls,
                self._playing_thresh,
                roi_frac=self._playing_roi,
            )
        else:
            ss_ok = False
            ss_val = 0.0
            ss_name = ""
            rs_ok = False
            rs_val = 0.0
            rs_name = ""
            pl_ok = False
            pl_val = 0.0
            pl_name = ""

        if ss_ok:
            target = STATE_SONG_SELECT
        elif rs_ok:
            target = STATE_RESULTS
        elif pl_ok:
            target = STATE_PLAYING
        elif drums_present:
            target = STATE_PLAYING
        else:
            target = STATE_OTHER

        prev_state = self._state
        if target != self._target_state:
            self._target_state = target
            self._state_streak = 1
        else:
            self._state_streak += 1

        if target != self._state and self._state_streak >= self._state_confirm_frames:
            self._state = target
            self._state_streak = 0

        state_transitioned = prev_state != self._state

        prev_armed = self._armed

        if drums_present:
            self._good_streak += 1
            self._bad_streak = 0
        else:
            self._bad_streak += 1
            self._good_streak = 0

        if self._armed:
            if self._bad_streak >= self.disarm_after_bad_frames:
                self._armed = False
        else:
            if self._good_streak >= self.arm_after_good_frames:
                self._armed = True

        armed_transitioned = prev_armed != self._armed

        info: dict[str, Any] = {
            "enabled": True,
            "state": self._state,
            "armed": self._armed,
            "transitioned": armed_transitioned,
            "state_transitioned": state_transitioned,
            "per_ok": per_ok,
            "lap_vars": lap_vars,
            "mean_grays": mean_grays,
            "good_streak": self._good_streak,
            "bad_streak": self._bad_streak,
            "song_select_tpl_ok": ss_ok,
            "song_select_tpl_val": ss_val,
            "song_select_tpl_name": ss_name,
            "results_tpl_ok": rs_ok,
            "results_tpl_val": rs_val,
            "results_tpl_name": rs_name,
            "playing_tpl_ok": pl_ok,
            "playing_tpl_val": pl_val,
            "playing_tpl_name": pl_name,
        }
        return self._state, info

    def _load_and_prepare_templates(
        self,
        kind: str,
    ) -> list[tuple[str, NDArray[np.uint8]]]:
        templates: list[tuple[str, NDArray[np.uint8]]] = []
        for name, tpl_path in list_scene_templates(kind):
            img = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
            if img is None:
                img_bytes = np.fromfile(str(tpl_path), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("无法读取场景子模块模板：%s", tpl_path)
                continue
            prepared = self._prepare_match_image(img)
            th, tw = prepared.shape[:2]
            logger.info("已加载场景子模块模板：%s/%s (%dx%d)", kind, name, tw, th)
            templates.append((name, prepared))
        if not templates:
            logger.debug("未找到场景子模块模板：%s", kind)
        return templates

    def _match_templates(
        self,
        frame_bgr: NDArray[np.uint8],
        templates: list[tuple[str, NDArray[np.uint8]]],
        threshold: float,
        *,
        roi_frac: list[float] | tuple[float, float, float, float] | None = None,
    ) -> tuple[bool, float, str]:
        if not templates:
            return False, 0.0, ""
        frame_roi = self._crop_roi(frame_bgr, roi_frac)
        if frame_roi is None:
            return False, 0.0, ""
        match_frame = self._prepare_match_image(frame_roi)
        fh, fw = match_frame.shape[:2]

        best_val = 0.0
        best_name = ""
        vote_count = 0
        required_votes = min(self._match_vote_min, len(templates))

        for name, tpl in templates:
            th, tw = tpl.shape[:2]
            if th > fh or tw > fw or th <= 0 or tw <= 0:
                continue
            result = cv2.matchTemplate(match_frame, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = float(max_val)
                best_name = name
            if max_val >= threshold:
                vote_count += 1

        ok = vote_count >= required_votes
        return ok, best_val, best_name

    def _prepare_match_image(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        out = img
        k = self._match_blur_ksize
        if k is not None and k >= 3:
            k = k if k % 2 == 1 else k + 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        scale = self._match_downscale
        if scale is not None and 0.1 <= scale < 1.0:
            h, w = out.shape[:2]
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            if nw != w or nh != h:
                out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        return out

    def _crop_roi(
        self,
        img: NDArray[np.uint8],
        roi_frac: list[float] | tuple[float, float, float, float] | None,
    ) -> NDArray[np.uint8] | None:
        if not roi_frac or len(roi_frac) != 4:
            return img
        h, w = img.shape[:2]
        x0 = int(max(0.0, min(1.0, float(roi_frac[0]))) * w)
        y0 = int(max(0.0, min(1.0, float(roi_frac[1]))) * h)
        x1 = int(max(0.0, min(1.0, float(roi_frac[2]))) * w)
        y1 = int(max(0.0, min(1.0, float(roi_frac[3]))) * h)
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]

    def _measure_patches(
        self,
        frame_bgr: NDArray[np.uint8],
        layout: LaneLayout,
    ) -> tuple[list[bool], list[float], list[float]]:
        h, w = layout.frame_h, layout.frame_w
        cy = int(self.drum_y_frac * h)
        half = max(4, int(self.patch_half_w_frac * w))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        per_ok: list[bool] = []
        lap_vars: list[float] = []
        mean_grays: list[float] = []

        for i in range(4):
            cx = layout.center_x[i]
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            if x1 <= x0 or y1 <= y0:
                lap_vars.append(0.0)
                mean_grays.append(0.0)
                per_ok.append(False)
                continue

            roi = gray[y0:y1, x0:x1]
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            var = float(lap.var())
            mg = float(np.mean(roi))
            lap_vars.append(var)
            mean_grays.append(mg)
            ok = (
                var >= self.min_laplace_var
                and self.min_mean_gray <= mg <= self.max_mean_gray
            )
            per_ok.append(ok)

        return per_ok, lap_vars, mean_grays
