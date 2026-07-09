# -*- coding: utf-8 -*-
import json

from maa.context import Context
from maa.custom_action import CustomAction

from .logger import logger


class UF_ActionLogger(CustomAction):
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        info_logger = json.loads(argv.custom_action_param).get("info")
        debug_logger = json.loads(argv.custom_action_param).get("debug")

        if info_logger:
            if isinstance(info_logger, str):
                info_logger = [info_logger]
            for info_one in info_logger:
                logger.info(f'{info_one}')
        if debug_logger:
            if isinstance(debug_logger, str):
                debug_logger = [debug_logger]
            for debug_one in debug_logger:
                logger.debug(f'{debug_one}')
        return True


class UF_ActionMoveScreen(CustomAction):
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:

        context_data = context
        argv_box = argv.box
        width, height = context.tasker.controller.resolution
        middle_x = int(width / 2)
        middle_y = int(height / 2)
        # new_image = context.tasker.controller.post_screencap().wait().get()
        target_box = None
        # a = argv.custom_action_param
        if argv.custom_action_param != 'null':
            target_box = json.loads(argv.custom_action_param).get("target_box", None)

        if target_box is None:
            x, y, w, h = argv.box
            target_x = int(x + w / 2)
            target_y = int(y + h / 2)
        else:
            if len(target_box) == 4:
                x, y, w, h = target_box
                target_x = int(x + w / 2)
                target_y = int(y + h / 2)
            elif len(target_box) == 2:
                x, y= target_box
                target_x = int(x)
                target_y = int(y)
            else:
                logger.error(f'error box: {target_box}')
                return True

        # context.tasker.controller.post_click(target_x, target_y)
        context.tasker.controller.post_swipe(middle_x ,middle_y,target_x, target_y,duration=10,pressure=0)
        print()

        return True
