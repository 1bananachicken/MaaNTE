import logging

from maa.context import Context


class ProgressReporter:
    """通用任务进度报告器，用于在循环执行任务时显示当前进度。"""

    def __init__(self, task_name: str, logger: logging.Logger):
        self.task_name = task_name
        self.logger = logger

    def report_loop(self, context: Context, loop_node: str, loop_total: int):
        """报告外层循环进度。

        Args:
            context: 任务上下文
            loop_node: 控制循环的 pipeline 节点名
            loop_total: 循环总次数，<=0 表示无限循环
        """
        current = context.get_hit_count(loop_node)
        if loop_total > 0:
            self.logger.info(
                f"[{self.task_name}] 循环进度: 第{current}/{loop_total}次"
            )
        else:
            self.logger.info(
                f"[{self.task_name}] 循环进度: 第{current}次"
            )

    def report_count(self, current: int, total: int):
        """报告内层执行进度。

        Args:
            current: 当前执行次数
            total: 总执行次数
        """
        self.logger.info(
            f"[{self.task_name}] 执行进度: {current}/{total}"
        )

    def report_combined(
        self,
        context: Context,
        loop_node: str,
        loop_total: int,
        count_current: int,
        count_total: int,
    ):
        """报告组合进度（外层循环 + 内层执行）。

        Args:
            context: 任务上下文
            loop_node: 控制循环的 pipeline 节点名
            loop_total: 循环总次数，<=0 表示无限循环
            count_current: 当前执行次数
            count_total: 总执行次数
        """
        loop_current = context.get_hit_count(loop_node)
        if loop_total > 0:
            self.logger.info(
                f"[{self.task_name}] 第{loop_current}/{loop_total}次循环, "
                f"第{count_current}/{count_total}次执行"
            )
        else:
            self.logger.info(
                f"[{self.task_name}] 第{loop_current}次循环, "
                f"第{count_current}/{count_total}次执行"
            )
