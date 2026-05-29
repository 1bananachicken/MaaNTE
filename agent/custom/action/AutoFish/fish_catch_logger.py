"""
钓鱼收益统计 Custom Action
- 在钓鱼结果界面截图，OCR 识别鱼名
- 模糊匹配价格表，容忍OCR错别字
- 查表获取贝壳价值，累计统计
- 通过 focus 推送到前端日志
"""

import json
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


# 价格表路径
_PRICE_TABLE_PATH = Path(__file__).parents[4] / "assets" / "resource" / "base" / "fish_price_table.json"
if not _PRICE_TABLE_PATH.exists():
    _PRICE_TABLE_PATH = Path(__file__).parents[4] / "resource" / "base" / "fish_price_table.json"

# 加载价格表
def _load_price_table() -> dict:
    """加载价格表，返回 {鱼名: avg价格} 的扁平dict"""
    if _PRICE_TABLE_PATH.exists():
        with open(_PRICE_TABLE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result = {}
        for name, val in raw.items():
            if isinstance(val, dict):
                result[name] = val.get("avg", 0)
            else:
                result[name] = val
        return result
    return {}


def _edit_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _fuzzy_match(ocr_text: str, name_list: list, max_dist: int = 2) -> str | None:
    """模糊匹配：在鱼名列表中找编辑距离最小的，超过阈值则返回None"""
    if not ocr_text:
        return None
    # 精确匹配优先
    if ocr_text in name_list:
        return ocr_text
    # 模糊匹配
    best_name = None
    best_dist = max_dist + 1
    for name in name_list:
        # 长度差太大直接跳过
        if abs(len(name) - len(ocr_text)) > max_dist:
            continue
        dist = _edit_distance(ocr_text, name)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name if best_dist <= max_dist else None


@AgentServer.custom_action("fish_catch_logger")
class FishCatchLogger(CustomAction):
    # 类变量：跨轮次累计
    _total_count: int = 0
    _total_shells: int = 0
    _catch_log: dict = {}  # {鱼名: 数量}
    _price_table: dict = _load_price_table()
    _initialized: bool = False  # 标记本轮是否已初始化

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        """在钓鱼结果界面，OCR识别鱼名并累计统计"""

        # 首次调用时 reset（确保每次启动任务从0开始）
        if not FishCatchLogger._initialized:
            FishCatchLogger.reset()
            FishCatchLogger._initialized = True

        # 截图
        image = context.tasker.controller.post_screencap().wait().get()

        # OCR 识别鱼名
        fish_name = self._recognize_fish_name(context, image)

        if fish_name:
            price = self._price_table.get(fish_name, 0)
            FishCatchLogger._total_count += 1
            FishCatchLogger._total_shells += price
            FishCatchLogger._catch_log[fish_name] = FishCatchLogger._catch_log.get(fish_name, 0) + 1
            msg = f"第{FishCatchLogger._total_count}条 {fish_name}，累计预期收益{FishCatchLogger._total_shells}贝壳"
        else:
            FishCatchLogger._total_count += 1
            msg = f"第{FishCatchLogger._total_count}条（识别失败），累计预期收益{FishCatchLogger._total_shells}贝壳"

        # 通过 focus 推送到前端
        try:
            context.override_pipeline({
                "FishCatchLogger_Notify": {
                    "recognition": "DirectHit",
                    "action": "DoNothing",
                    "focus": {
                        "Node.Action.Succeeded": {
                            "content": msg,
                            "display": ["log", "toast"]
                        }
                    }
                }
            })
            context.run_task("FishCatchLogger_Notify")
        except Exception as e:
            print(f"[FishLog] focus推送失败: {e}")

        # 按 Esc 关闭结果界面
        context.tasker.controller.post_click_key(27).wait()

        return CustomAction.RunResult(success=True)

    def _recognize_fish_name(self, context: Context, image) -> str:
        """OCR 识别鱼名，并模糊匹配到价格表"""
        reco_detail = context.run_recognition(
            "FishCatchLogger_OCR_FishName",
            image,
            pipeline_override={
                "FishCatchLogger_OCR_FishName": {
                    "recognition": "OCR",
                    "roi": [440, 120, 400, 50],
                    "expected": [],
                    "only_rec": False
                }
            }
        )

        if reco_detail and reco_detail.all_results:
            best = reco_detail.all_results[0]
            ocr_text = best.text.strip() if hasattr(best, 'text') else ""
            if not ocr_text:
                return None
            # 模糊匹配价格表中的鱼名
            matched = _fuzzy_match(ocr_text, list(self._price_table.keys()))
            return matched
        return None

    @classmethod
    def get_summary(cls) -> str:
        """获取当前累计汇总文本"""
        if cls._total_count == 0:
            return "本轮未钓到鱼"
        summary = f"🎣 本轮钓鱼: {cls._total_count}条"
        if cls._total_shells > 0:
            summary += f" | 预计收益: {cls._total_shells}贝壳"
        return summary

    @classmethod
    def reset(cls):
        """重置统计"""
        cls._total_count = 0
        cls._total_shells = 0
        cls._catch_log = {}


@AgentServer.custom_action("fish_catch_summary")
class FishCatchSummary(CustomAction):
    """任务结束时上报钓鱼收益汇总到前端"""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        summary = FishCatchLogger.get_summary()

        # 推送汇总到前端
        try:
            context.override_pipeline({
                "FishCatchSummary_Notify": {
                    "recognition": "DirectHit",
                    "action": "DoNothing",
                    "focus": {
                        "Node.Action.Succeeded": {
                            "content": summary,
                            "display": ["log", "toast"]
                        }
                    }
                }
            })
            context.run_task("FishCatchSummary_Notify")
        except Exception as e:
            print(f"[FishLog] 汇总推送失败: {e}")

        # 重置计数（为下一轮准备）
        FishCatchLogger.reset()
        FishCatchLogger._initialized = False

        return CustomAction.RunResult(success=True)
