# -*- coding: utf-8 -*-
"""
动态 OCR 自定义识别 — 从 i18n_ocr_data 取多语言 expected 文本，调用 OCR。

与 i18n_ocr_data.py 的关系:
    i18n_ocr_data.py  →  纯数据（I18N_OCR dict + get_expected）
    dynamic_ocr.py    →  识别逻辑（CustomRecognition，从 dict 读 expected → 调 OCR）

用法 (pipeline JSON):
    {
        "recognition": "Custom",
        "custom_recognition": "dynamic_ocr",
        "custom_recognition_param": "{\"node\": \"FurnitureHamsterBall\"}",
        "roi": [948, 170, 308, 517]
    }

也可以在 CustomAction 中直接调用 i18n_ocr_data.get_expected() +
context.run_recognition_direct()。
"""

from __future__ import annotations

import json
from typing import Optional

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR, JRecognitionType

from utils.i18n_ocr_data import I18N_OCR, TL
from utils.logger import logger


@AgentServer.custom_recognition("dynamic_ocr")
class DynamicOCR(CustomRecognition):
    """根据 node 名从 I18N_OCR dict 取对应语言的 expected 值，执行 OCR。

    custom_recognition_param:
        {"node": "FurnitureHamsterBall"}       # 必选
        {"node": "...", "threshold": 0.7}      # 可选，默认 0.3
        {"node": "...", "lang": "ja_jp"}       # 可选，默认自动检测
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params = self._parse_param(argv.custom_recognition_param)
        node = params.get("node", "")

        expected = TL(node, params.get("lang"))
        if not expected:
            logger.warning("dynamic_ocr: no expected text for node=%s", node)
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        threshold = float(params.get("threshold", 0.3))
        result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(
                roi=(argv.roi.x, argv.roi.y, argv.roi.w, argv.roi.h),
                expected=[expected],
                threshold=threshold,
            ),
            argv.image,
        )

        if result and result.hit and result.box:
            return CustomRecognition.AnalyzeResult(
                box=(result.box.x, result.box.y, result.box.w, result.box.h),
                detail={"node": node, "expected": expected},
            )
        return CustomRecognition.AnalyzeResult(box=None, detail={})

    @staticmethod
    def _parse_param(raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
