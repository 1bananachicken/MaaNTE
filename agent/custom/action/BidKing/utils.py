"""
拍卖之王 - 共享小工具

入口动作（place_bid）与各策略都用到的东西放这里：
  - OCR 读取指定区域（含多级兜底提取）
  - 价格文本解析（整串匹配，规则见 openspec 的 bidking-bidding 能力契约）
  - 图像裁剪、配置值转换
"""

import re
from typing import Optional

from maa.context import Context
from maa.pipeline import JRecognitionType, JOCR

from ..Common.utils import get_image
from utils.logger import logger


# 全角数字与常见全角符号 -> 半角（OCR 可能返回全角）
_FULLWIDTH = {ord("０") + i: ord("0") + i for i in range(10)}
_FULLWIDTH.update({ord("，"): ord(","), ord("．"): ord("."), ord("　"): ord(" ")})

# 价格两端可能出现的噪声字符
_TRIM_CHARS = "¥$￥|:：。. "

# 只接受两种整串写法：
#   1) 千位分隔写法：1,222,418 / 14.346（OCR 把分隔符读成小数点）
#   2) 纯数字：0 / 91 / 839（小额估价游戏不加分隔符）
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[,.]\d{3})+")
_PLAIN_RE = re.compile(r"\d{1,6}")


def extract_price(text: Optional[str]) -> Optional[int]:
    """
    从 OCR 文本中解析价格；不符合写法时返回 None（不做猜测）。

    整串匹配很关键：`可输入范围0~2,524,741`、`1.23M` 这类文本必须失败，
    否则会把提示文字或紧凑写法读成余额/错误金额。
    """
    if not text:
        return None

    cleaned = text.translate(_FULLWIDTH).strip(_TRIM_CHARS).strip()

    if _THOUSANDS_RE.fullmatch(cleaned):
        return int(cleaned.replace(",", "").replace(".", ""))
    if _PLAIN_RE.fullmatch(cleaned):
        return int(cleaned)
    return None


def crop(img, roi):
    """按 [x, y, w, h] 裁剪图像，越界自动收敛"""
    x, y, w, h = [int(v) for v in roi]
    height, width = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def to_positive_int(value) -> Optional[int]:
    """把配置值转成正整数；非法（非数字、0、负数）返回 None"""
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def ocr_text(context: Context, img, roi, debug: bool = False) -> Optional[str]:
    """对给定图像做 OCR 并提取文本（沿用项目统一调用方式）"""
    try:
        detail = context.run_recognition_direct(
            JRecognitionType.OCR, JOCR(roi=roi), img
        )
    except Exception as e:
        logger.error(f"❌ OCR 调用异常: {e}")
        return None

    if detail is None:
        logger.warning("⚠️ OCR 返回 None")
        return None

    if debug:
        logger.debug(f"OCR detail 类型={type(detail)}, 内容={detail}")

    best = getattr(detail, "best_result", None)
    if best is not None:
        text = getattr(best, "text", None)
        if text:
            return text.strip()

    all_results = getattr(detail, "all_results", None)
    if all_results:
        joined = " ".join(r.text for r in all_results if getattr(r, "text", None))
        if joined.strip():
            return joined.strip()

    if isinstance(detail, str):
        return detail.strip()

    text = getattr(detail, "text", None)
    if text:
        return text.strip()

    logger.warning(f"⚠️ 无法从 OCR detail 提取文本，类型={type(detail)}")
    return None


def read_region_text(
    context: Context, controller, roi, debug: bool = False
) -> Optional[str]:
    """截图并 OCR 指定区域"""
    img = get_image(controller)
    if img is None:
        logger.error("❌ 截图失败")
        return None
    return ocr_text(context, img, roi, debug)
