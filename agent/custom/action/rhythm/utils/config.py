from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "lanes": {
        "center_x_frac": [0.225, 0.406, 0.596, 0.771],
        "top_center_x_frac": [0.214, 0.406, 0.596, 0.783],
        "half_width_frac": 0.028,
        "judge_line_y_frac": 0.78,
        "judge_line_y_frac_by_lane": [0.78, 0.78, 0.78, 0.78],
        "judge_band_half_height_frac": 0.03,
    },
    "template_detection": {
        "thresholds": [0.81, 0.80, 0.80, 0.81],
        "cooldown_sec": 0.05,
        "cooldown_sec_by_lane": [0.05, 0.05, 0.05, 0.05],
        "region_extend_up_frac": 0.14,
        "region_extend_down_frac": 0.15,
        "region_width_multiplier": 4.0,
        "enabled_lanes": [True, True, True, True],
    },
    "scene": {
        "state_confirm_frames": 1,
        "song_select_match_threshold": 0.75,
        "results_match_threshold": 0.75,
        "playing_match_threshold": 0.75,
        "match_vote_min": 1,
        "playing_check_interval": 5,
        "song_select_to_playing_lock_sec": 8.0,
        "template_roi": {
            "song_select": {
                "logo": [0, 0, 80, 80],
                "level": [1013, 184, 87, 58],
                "start": [925, 641, 279, 44]
            },
            "playing": {
                "pause": [0, 0, 96, 96],
                "rate": [1076, 0, 95, 171],
                "score": [1076, 0, 95, 171]
            },
            "results": {
                "max_combo": [241, 496, 800, 70],
                "rate": [241, 496, 800, 70],
                "score": [241, 496, 800, 70]
            }
        },
    },
    "song_select": {
        "enabled": False,
        "song_name": "",
        "song_list_roi": [47, 117, 550, 510],
        "scroll_area_x_frac": 0.25,
        "scroll_area_y_frac": 0.50,
        "scroll_delta": -1,
        "max_scroll_attempts": 30,
        "match_threshold": 0.75,
        "start_match_threshold": 0.75,
        "click_delay_sec": 0.5,
        "start_delay_sec": 0.8,
        "scroll_settle_delay_sec": 0.4,
        "click_reverify_threshold": 0.70,
        "max_click_reverify_retries": 2,
    },
    "auto_repeat": {
        "enabled": False,
        "count": 5,
        "dismiss_delay_sec": 0.8,
    },
    "vitality_detect": {
        "enabled": True,
        "roi": [544, 622, 184, 46],
        "cost_pattern": "(\\d+)",
        "min_confirm_reads": 2,
        "confirm_interval_sec": 0.3,
        "vitality_threshold": 1,
    },
    "keys": {
        "press_delay_sec": 0.0,
        "press_delay_sec_by_lane": [0.0, 0.0, 0.0, 0.0],
        "key_hold_sec": 0.01,
    },
    "run": {
        "target_fps": 60,
    },
}


def load_rhythm_config() -> dict[str, Any]:
    here = Path(__file__).resolve()
    cfg_paths: list[Path] = []
    for i in range(len(here.parents)):
        root = here.parents[i]
        cfg_paths.append(root / "resource" / "base" / "rhythm_config.json")
        cfg_paths.append(root / "assets" / "resource" / "base" / "rhythm_config.json")
    for p in cfg_paths:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                logger.info("已加载演奏配置文件: %s", p)
                return loaded
            except Exception:
                logger.warning("演奏配置文件读取失败: %s，使用内置默认值", p)
    logger.info("未找到外部演奏配置文件，使用内置默认值")
    return dict(_DEFAULT_CONFIG)
