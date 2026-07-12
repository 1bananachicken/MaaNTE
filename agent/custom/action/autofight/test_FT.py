# -*- coding: utf-8 -*-


import json
import time
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from .logger import logger
from maa.agent.agent_server import AgentServer

@AgentServer.custom_recognition("FT_StopFight")
class FT_StopFight(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')

        start_FT_time = json.loads(argv.custom_recognition_param).get("start_time" , -1)
        fight_spend_time = json.loads(argv.custom_recognition_param).get("fight_spend_time", 50)

        if start_FT_time == -1:
            current_time = time.time()
            current_node = argv.node_name
            context.override_pipeline({
                current_node: {
                    "custom_recognition_param": {
                        "start_time": str(current_time),
                        "fight_spend_time": int(fight_spend_time)

                    }}})
            logger.debug(f'开始时间:{current_time},运行{fight_spend_time}秒')
            return None
        elif time.time() > float(start_FT_time) + float(fight_spend_time):
            logger.debug(f'到达运行时间,已运行运行{time.time() - float(start_FT_time)}秒')
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")
        else:
            return None