"""
粉爪大劫案 收益统计
- 撤离成功时累计方斯和粉爪币
- 撤离失败不计
- 通过 focus 推送到前端
"""

import re

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

# 每次撤离成功的固定收益（可根据实际调整）
REWARD_PER_RUN = {
    "方斯": 0,  # 后续根据实际 OCR 或固定值填入
    "粉爪币": 0,  # 后续根据实际 OCR 或固定值填入
}

REWARD_OCR_ROIS: dict[str, tuple[int, int, int, int]] = {
    "方斯": (684, 238, 112, 20),
    "粉爪币": (800, 409, 88, 25),
}

_AMOUNT_LIKE_RE = re.compile(r"[0-9OoQqIl|!SsBbZz,，.￥¥$₩€£¢₽₹₫₱¤＋+\-]+")
_OCR_DIGIT_TABLE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "b": "8",
        "Z": "2",
        "z": "2",
    }
)
_AMOUNT_PREFIX_CHARS = {"x", "X"}


class PinkPawRewardTracker:
    """全局收益追踪器（类似 FishCatchLogger 的类变量模式）"""

    _success_count: int = 0
    _fail_count: int = 0
    _total_fansi: int = 0
    _total_pinkcoins: int = 0
    _initialized: bool = False

    @classmethod
    def reset(cls):
        cls._success_count = 0
        cls._fail_count = 0
        cls._total_fansi = 0
        cls._total_pinkcoins = 0

    @classmethod
    def on_evacuate_success(cls, fansi: int = 0, pinkcoins: int = 0):
        """撤离成功时调用"""
        cls._success_count += 1
        cls._total_fansi += fansi
        cls._total_pinkcoins += pinkcoins

    @classmethod
    def on_evacuate_fail(cls):
        """撤离失败时调用"""
        cls._fail_count += 1

    @classmethod
    def get_msg(cls) -> str:
        """获取当前状态的一行消息"""
        msg = f"第{cls._success_count + cls._fail_count}局"
        if cls._success_count > 0 or cls._fail_count > 0:
            msg += f"（成功{cls._success_count}/失败{cls._fail_count}）"
        if cls._total_fansi > 0:
            msg += f"，累计方斯{cls._total_fansi}"
        if cls._total_pinkcoins > 0:
            msg += f"，累计粉爪币{cls._total_pinkcoins}"
        return msg

    @classmethod
    def get_summary(cls) -> str:
        parts = [f"🐾 粉爪大劫案: {cls._success_count}局成功/{cls._fail_count}局失败"]
        if cls._total_fansi > 0:
            parts.append(f"方斯{cls._total_fansi}")
        if cls._total_pinkcoins > 0:
            parts.append(f"粉爪币{cls._total_pinkcoins}")
        return " | ".join(parts)


def _result_text(result) -> str:
    if result is None:
        return ""

    texts: list[str] = []
    seen: set[str] = set()

    def append_text(item):
        if item is None:
            return
        text = item.text if hasattr(item, "text") else str(item)
        text = text.strip()
        if not text or text in seen:
            return
        seen.add(text)
        texts.append(text)

    all_results = getattr(result, "all_results", None) or []
    for item in all_results:
        append_text(item)

    return " ".join(texts)


def _normalize_frame(image):
    if image is None:
        return image
    if not hasattr(image, "shape") or len(image.shape) != 3 or image.shape[2] != 4:
        return image
    try:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    except Exception:
        return image


def _parse_amount(text: str) -> int:
    amounts = []
    for match in _AMOUNT_LIKE_RE.finditer(text or ""):
        raw = match.group(0)
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if before.isascii() and before.isalpha() and before not in _AMOUNT_PREFIX_CHARS:
            continue
        if after.isascii() and after.isalpha():
            continue

        normalized = raw.translate(_OCR_DIGIT_TABLE)
        digits = re.sub(r"\D", "", normalized)
        if not any(ch.isdigit() for ch in raw) and len(digits) < 2:
            continue
        if digits:
            amounts.append(int(digits))
    return amounts[-1] if amounts else 0


def _ocr_amount(context: Context, image, label: str) -> int:
    try:
        roi = REWARD_OCR_ROIS[label]
        result = context.run_recognition_direct(
            JRecognitionType.OCR, JOCR(roi=roi), _normalize_frame(image)
        )
    except Exception as exc:
        print(f"[PinkPawReward] OCR {label} failed: {exc}")
        return 0

    text = _result_text(result)
    amount = _parse_amount(text)
    if amount <= 0:
        print(f"[PinkPawReward] OCR {label} no amount, text: {text!r}")
    return amount


def _ocr_reward_amounts(context: Context) -> tuple[int, int]:
    controller = getattr(getattr(context, "tasker", None), "controller", None)
    if controller is None:
        return 0, 0

    try:
        image = controller.post_screencap().wait().get()
    except Exception as exc:
        print(f"[PinkPawReward] screencap failed: {exc}")
        return 0, 0

    fansi = _ocr_amount(context, image, "方斯")
    pinkcoins = _ocr_amount(context, image, "粉爪币")
    return fansi, pinkcoins


def notify_pinkpaw_reward(
    context: Context, success: bool, fansi: int = 0, pinkcoins: int = 0
):
    """
    在 pinkpaw_core1/core2 中撤离后调用此函数推送收益。
    success=True 表示撤离成功，False 表示失败。
    """
    if not PinkPawRewardTracker._initialized:
        PinkPawRewardTracker.reset()
        PinkPawRewardTracker._initialized = True

    if success:
        if fansi <= 0 or pinkcoins <= 0:
            ocr_fansi, ocr_pinkcoins = _ocr_reward_amounts(context)
            if fansi <= 0:
                fansi = ocr_fansi or REWARD_PER_RUN["方斯"]
            if pinkcoins <= 0:
                pinkcoins = ocr_pinkcoins or REWARD_PER_RUN["粉爪币"]

        PinkPawRewardTracker.on_evacuate_success(fansi, pinkcoins)
        current_parts = []
        if fansi > 0:
            current_parts.append(f"方斯+{fansi}")
        if pinkcoins > 0:
            current_parts.append(f"粉爪币+{pinkcoins}")
        current_msg = f"本局{'，'.join(current_parts)}；" if current_parts else ""
        msg = f"✅ 撤离成功！{current_msg}{PinkPawRewardTracker.get_msg()}"
    else:
        PinkPawRewardTracker.on_evacuate_fail()
        msg = f"❌ 撤离失败。{PinkPawRewardTracker.get_msg()}"

    try:
        context.override_pipeline(
            {
                "PinkPawReward_Notify": {
                    "recognition": "DirectHit",
                    "action": "DoNothing",
                    "focus": {"Node.Action.Starting": msg},
                }
            }
        )
        context.run_task("PinkPawReward_Notify")
    except Exception:
        pass


@AgentServer.custom_action("pinkpaw_reward_summary")
class PinkPawRewardSummary(CustomAction):
    """任务完全结束时上报汇总"""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        summary = PinkPawRewardTracker.get_summary()
        try:
            context.override_pipeline(
                {
                    "PinkPawReward_Summary": {
                        "recognition": "DirectHit",
                        "action": "DoNothing",
                        "focus": {"Node.Action.Starting": summary},
                    }
                }
            )
            context.run_task("PinkPawReward_Summary")
        except Exception:
            pass

        PinkPawRewardTracker.reset()
        PinkPawRewardTracker._initialized = False
        return CustomAction.RunResult(success=True)
