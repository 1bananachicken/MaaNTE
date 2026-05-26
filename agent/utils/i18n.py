"""i18n 翻译模块。加载 locales/ 下的语言 JSON，提供 T() 查 key 和 RenderHTML()。"""

import html
import json
import os
from pathlib import Path
from typing import Any

from utils import pienv
from utils.logger import logger

LangZhCN = "zh_cn"
LangZhTW = "zh_tw"
LangEnUS = "en_us"
LangJaJP = "ja_jp"
LangKoKR = "ko_kr"
DefaultLang = LangZhCN

_messages: dict[str, str] = {}
_current_lang: str = DefaultLang
_locale_dir: str = ""


def _normalize_lang(s: str) -> str:
    s = s.strip().lower()
    if s in (LangZhCN, LangZhTW, LangEnUS, LangJaJP, LangKoKR):
        return s
    return DefaultLang


def _resolve_locale_dir() -> str:
    """从 cwd 向上搜索 resource/locales/agent/ 目录。"""
    candidates = []
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parent.parent.parent)
    except Exception:
        pass

    for base in candidates:
        d = base
        for _ in range(6):
            for rel in (
                "assets/resource/locales/agent",
                "resource/locales/agent",
            ):
                candidate = d / rel
                if (candidate / f"{DefaultLang}.json").exists():
                    return str(candidate)
            if d.parent == d:
                break
            d = d.parent

    return str(Path.cwd() / "assets" / "resource" / "locales" / "agent")


def _load_messages(lang: str) -> dict[str, str]:
    """先加载默认语言，再叠加目标语言（缺失 key 回退到默认语言）。"""
    msgs: dict[str, str] = {}

    def load(l: str) -> bool:
        path = os.path.join(_locale_dir, f"{l}.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                msgs.update(data)
            return True
        except Exception as e:
            logger.warning(f"加载 i18n 文件失败: {path} — {e}")
            return False

    default_ok = load(DefaultLang)
    if lang != DefaultLang:
        load(lang)
        if not default_ok:
            return {}
    elif not default_ok:
        return {}
    return msgs


def init():
    """初始化 i18n：检测语言 → 加载翻译 JSON。在 main.py agent() 中 AgentServer.start_up() 前调用。"""
    global _messages, _current_lang, _locale_dir

    raw = pienv.client_language()
    lang = _normalize_lang(raw) if raw else DefaultLang
    _locale_dir = _resolve_locale_dir()
    _messages = _load_messages(lang)
    _current_lang = lang

    logger.info(
        f"i18n initialized: PI_CLIENT_LANGUAGE={raw}, "
        f"resolved_lang={lang}, locale_dir={_locale_dir}, "
        f"message_count={len(_messages)}"
    )


def lang() -> str:
    return _current_lang


def T(key: str, *args) -> str:
    """返回当前语言的翻译文本。args 非空时做 %-格式化。"""
    val = _messages.get(key)
    if val is None:
        return key
    if args:
        return val % args
    return val


def separator() -> str:
    """CJK 用 、，英文/韩文用 ,"""
    return ", " if _current_lang in (LangEnUS, LangKoKR) else "、"


def RenderHTML(key: str, data: dict[str, Any] | None = None) -> str:
    """渲染注册过的 HTML 模板。当前简化实现，直接返回 T(key)。"""
    return T(key)
