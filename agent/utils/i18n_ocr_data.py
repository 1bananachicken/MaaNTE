# -*- coding: utf-8 -*-
"""
OCR 多语言文本映射 — 纯数据，无逻辑。

集中管理所有 OCR expected 文本的多语言版本。
Pipeline JSON / CustomAction 通过 node 名查找对应语言的 expected 值。

用法:
    from utils.i18n_ocr_data import I18N_OCR, TL

    expected = I18N_OCR["FurnitureHamsterBall"]["ja_jp"]  # 直接取
    expected = TL("FurnitureHamsterBall")                  # 自动根据当前语言取

扩展:
    添加新节点只需在本文件追加一条 dict 条目，不改任何 JSON。
"""

from __future__ import annotations

from typing import Optional

I18N_OCR: dict[str, dict[str, str]] = {
    # ── 家具 ──
    "FurnitureHamsterBall": {
        "zh_cn": "仓鼠球",
        "zh_tw": "倉鼠球",
        "en_us": r"(?i)Hamster\s*Ball",
        "ja_jp": "ハムスターボール",
    },
    "FurnitureFluff": {
        "zh_cn": "棉棉",
        "en_us": r"(?i)Fluff",
        "ja_jp": "モフンモフン",
    },
    "FurnitureDamagedCrate": {
        "zh_cn": "破损的木箱",
        "zh_tw": "破損的木箱",
        "en_us": r"(?i)Damaged\s*Crate",
        "ja_jp": "壊れた木箱",
    },
    "FurnitureIntactCrate": {
        "zh_cn": "完整的木箱",
        "en_us": r"(?i)Intact\s*Crate",
        "ja_jp": "完全な木箱",
    },
    # ── 房产 ──
    "FurnitureWienerApartments": {
        "zh_cn": "维纳公寓",
        "zh_tw": "維納公寓",
        "en_us": r"(?i)Wiener\s*Apartments",
        "ja_jp": "メゾンVINA",
    },
    "FurnitureEdenApartments": {
        "zh_cn": "伊登公寓",
        "en_us": r"(?i)Eden\s*Apartments",
        "ja_jp": "レジスEDEN",
    },
    "FurnitureSkyviewHalls": {
        "zh_cn": "天景空馆",
        "zh_tw": "天景空館",
        "en_us": r"(?i)Skyview\s*Halls",
        "ja_jp": "天景の館",
    },
    "FurnitureGoldenCapital": {
        "zh_cn": "金都云邸",
        "zh_tw": "金都雲邸",
        "en_us": r"(?i)Golden\s*Capital",
        "ja_jp": "パレス雲都",
    },
    "FurnitureTianJun": {
        "zh_cn": "天骏公阁",
    },
    "FurnitureFenglinVilla": {
        "zh_cn": "峰林别墅",
        "zh_tw": "峰林別墅",
        "en_us": r"(?i)Fenglin\s*Villa",
        "ja_jp": "シルヴァ邸苑",
        "ko_kr": "봉우리 별장",
    },
    # ═══════════════════════════════════
    # 继续在下面添加...
    # ═══════════════════════════════════
}

_DEFAULT_LANG = "zh_cn"


def TL(node: str, lang: Optional[str] = None) -> Optional[str]:
    """根据当前语言取对应节点的 OCR expected 文本。

    Args:
        node: I18N_OCR 中的 key
        lang: 语言代码，None 则自动从 PI 环境变量获取
    """
    if node not in I18N_OCR:
        return None
    translations = I18N_OCR[node]
    if lang is None:
        lang = _current_lang()
    return translations.get(lang) or translations.get(_DEFAULT_LANG)


def _current_lang() -> str:
    try:
        from utils.pienv import client_language
        lang = client_language()
        if lang:
            return lang
    except ImportError:
        pass
    return _DEFAULT_LANG
