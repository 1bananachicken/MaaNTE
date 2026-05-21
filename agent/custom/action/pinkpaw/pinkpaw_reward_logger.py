"""
粉爪大劫案 收益统计
- 撤离成功时累计方斯和粉粉币
- 撤离失败不计
- 通过 focus 推送到前端
"""

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


# 每次撤离成功的固定收益（可根据实际调整）
REWARD_PER_RUN = {
    "方斯": 0,       # 后续根据实际 OCR 或固定值填入
    "粉粉币": 0,    # 后续根据实际 OCR 或固定值填入
}


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
            msg += f"，累计粉粉币{cls._total_pinkcoins}"
        return msg

    @classmethod
    def get_summary(cls) -> str:
        parts = [f"🐾 粉爪大劫案: {cls._success_count}局成功/{cls._fail_count}局失败"]
        if cls._total_fansi > 0:
            parts.append(f"方斯{cls._total_fansi}")
        if cls._total_pinkcoins > 0:
            parts.append(f"粉粉币{cls._total_pinkcoins}")
        return " | ".join(parts)


def notify_pinkpaw_reward(context: Context, success: bool, fansi: int = 0, pinkcoins: int = 0):
    """
    在 pinkpaw_core1/core2 中撤离后调用此函数推送收益。
    success=True 表示撤离成功，False 表示失败。
    """
    if not PinkPawRewardTracker._initialized:
        PinkPawRewardTracker.reset()
        PinkPawRewardTracker._initialized = True

    if success:
        PinkPawRewardTracker.on_evacuate_success(fansi, pinkcoins)
        msg = f"✅ 撤离成功！{PinkPawRewardTracker.get_msg()}"
    else:
        PinkPawRewardTracker.on_evacuate_fail()
        msg = f"❌ 撤离失败。{PinkPawRewardTracker.get_msg()}"

    try:
        context.override_pipeline({
            "PinkPawReward_Notify": {
                "recognition": "DirectHit",
                "action": "DoNothing",
                "focus": {
                    "Node.Action.Starting": msg
                }
            }
        })
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
            context.override_pipeline({
                "PinkPawReward_Summary": {
                    "recognition": "DirectHit",
                    "action": "DoNothing",
                    "focus": {
                        "Node.Action.Starting": summary
                    }
                }
            })
            context.run_task("PinkPawReward_Summary")
        except Exception:
            pass

        PinkPawRewardTracker.reset()
        PinkPawRewardTracker._initialized = False
        return CustomAction.RunResult(success=True)
