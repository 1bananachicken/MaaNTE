import datetime
import functools
import time
import traceback
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, override

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.define import OCRResult
from utils.py_tools import type_match
from utils.virtual_key import Win_virtual_key

from ..Common.logger import get_logger

if TYPE_CHECKING:
    import numpy as np
    from maa.context import Context
    from maa.controller import Controller

logger = get_logger(__name__)

溜鱼_MAX_RANGE: Final = 120
溜鱼_GY_ROI: Final = [404, 44, 478, 12]


class Manual_stop(BaseException):
    pass


type Img = np.typing.NDArray


def get_img(controller: Controller, /) -> Img:
    """每次获取新截图要使用新的 controller"""
    start_time = time.time()
    while True:
        try:
            img = controller.post_screencap().get(wait=True)
        except Exception as e:
            logger.debug(e)
            if time.time() - start_time > 10:
                logger.error("获取截图超时")
                raise
            time.sleep(0.5)
            continue
        return img


@dataclass(kw_only=True, slots=True)
class Fish_option:
    终止时间: datetime.datetime | None

    溜鱼_midpoint_pix_range: int
    溜鱼_midpoint_sleep_time: int

    卖鱼买换饵开关: bool
    买饵次数: int


def get_option(context: Context, argv: CustomAction.RunArg, /) -> Fish_option:
    node_obj = context.get_node_object(argv.node_name)
    if node_obj is None:
        logger.error("获取 node_object 失败")
        raise RuntimeError
    attach = node_obj.attach
    del context, argv

    try:
        # 通用
        终止时间 = attach.get("终止时间")
        if not 终止时间:
            终止时间 = None
        else:
            if not isinstance(终止时间, str):
                raise TypeError("终止时间 类型不是 str")
            try:
                try:
                    终止时间 = datetime.datetime.fromisoformat(终止时间)
                except ValueError:
                    终止时间 = datetime.datetime.combine(
                        datetime.date.today(), datetime.time.fromisoformat(终止时间)
                    )
                if 终止时间.tzinfo is None:
                    终止时间 = 终止时间.astimezone()
                if 终止时间 < datetime.datetime.now().astimezone():
                    终止时间 += datetime.timedelta(days=1)
            except Exception as e:
                raise ValueError("格式化日期时间字符串错误") from e
            logger.info(f"终止时间: {终止时间}")

        # 溜鱼
        溜鱼_midpoint_pix_range = attach.get("溜鱼_midpoint_pix_range")
        溜鱼_midpoint_sleep_time = attach.get("溜鱼_midpoint_sleep_time")

        if 溜鱼_midpoint_pix_range is None:
            raise ValueError("溜鱼_midpoint_pix_range 必须填值")
        if 溜鱼_midpoint_sleep_time is None:
            raise ValueError("溜鱼_midpoint_sleep_time 必须填值")

        if not isinstance(溜鱼_midpoint_pix_range, int):
            raise TypeError("溜鱼_midpoint_pix_range 类型不是 int")
        if not isinstance(溜鱼_midpoint_sleep_time, int):
            raise TypeError("溜鱼_midpoint_sleep_time 类型不是 int")

        if not 0 < 溜鱼_midpoint_pix_range < 50:
            raise ValueError("溜鱼_midpoint_pix_range 值范围不正常")
        if not 0 <= 溜鱼_midpoint_sleep_time < 200:
            raise ValueError("溜鱼_midpoint_sleep_time 值范围不正常")

        # 卖鱼买换饵
        卖鱼买换饵开关 = attach.get("卖鱼买换饵开关")
        if 卖鱼买换饵开关 is None:
            raise ValueError("卖鱼买换饵开关 必须填值")
        if not isinstance(卖鱼买换饵开关, bool):
            raise TypeError("卖鱼买换饵开关 类型不是 bool")

        买饵次数 = attach.get("买饵次数")
        if 买饵次数 is None:
            raise ValueError("买饵次数 必须填值")
        买饵次数 = int(买饵次数)
        if not 0 < 买饵次数 < 20:
            raise ValueError("买饵次数 值范围不正常")

    except Exception as e:
        logger.error(f"选项初始化错误: {e} {e!r}\n{traceback.format_exc()}")
        raise

    return Fish_option(
        终止时间=终止时间,
        溜鱼_midpoint_pix_range=溜鱼_midpoint_pix_range,
        溜鱼_midpoint_sleep_time=溜鱼_midpoint_sleep_time,
        卖鱼买换饵开关=卖鱼买换饵开关,
        买饵次数=买饵次数,
    )


def reco_钓鱼按钮(context: Context, img: Img, /) -> bool:
    reco_detail = context.run_recognition(
        reco_钓鱼按钮.__name__,
        img,
        pipeline_override={
            reco_钓鱼按钮.__name__: {
                "recognition": {
                    "type": "Or",
                    "param": {
                        "any_of": [
                            {
                                "recognition": {
                                    "type": "FeatureMatch",
                                    "param": {
                                        "roi": [1144, 606, 84, 96],
                                        "template": "Fish/钓鱼按钮.png",
                                        "ratio": 0.88,
                                    },
                                }
                            },
                            {
                                "recognition": {
                                    "type": "OCR",
                                    "param": {
                                        "roi": [900, 670, 320, 32],
                                        "expected": ["R", "Q", "E", "F"],
                                    },
                                }
                            },
                        ]
                    },
                },
            }
        },
    )
    return not (reco_detail is None or reco_detail.box is None)


def reco_满舱_or_无饵(
    context: Context, img: Img, /
) -> Literal["渔获已满", "鱼饵用完"] | None:
    """:return: 返回结果，识别不到或识别错误返回 None"""
    reco_detail = context.run_recognition(
        reco_满舱_or_无饵.__name__,
        img,
        pipeline_override={
            reco_满舱_or_无饵.__name__: {
                "recognition": {
                    "type": "OCR",
                    "param": {"roi": [420, 342, 440, 36]},
                },
            }
        },
    )
    if reco_detail is None or reco_detail.box is None:
        return None

    all_results = reco_detail.all_results
    if not type_match(all_results, Sequence[OCRResult]):
        logger.error(
            f"{reco_满舱_or_无饵.__name__} OCR 结果不是 Sequence[OCRResult] 类型"
        )
        return None
    for res in all_results:
        if res.score < 0.9:
            continue
        if "需要装备鱼饵才可以钓鱼" in res.text:
            return "鱼饵用完"
        if "鱼舱中渔获已满" in res.text:
            return "渔获已满"
    return None


def reco_上鱼(context: Context, img: Img, /) -> bool:
    reco_detail = context.run_recognition(
        reco_上鱼.__name__,
        img,
        pipeline_override={
            reco_上鱼.__name__: {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [312, 82, 80, 28],
                        "expected": "鱼耐力",
                    },
                },
            }
        },
    )
    return not (reco_detail is None or reco_detail.box is None)


def reco_获鱼(context: Context, img: Img, /) -> bool:
    reco_detail = context.run_recognition(
        reco_获鱼.__name__,
        img,
        pipeline_override={
            reco_获鱼.__name__: {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [550, 630, 180, 46],
                        "expected": "点击空白区域关闭",
                    },
                },
            }
        },
    )
    return not (reco_detail is None or reco_detail.box is None)


def 溜鱼(context: Context, option: Fish_option, /) -> None:
    controller = context.tasker.controller

    for _ in range(溜鱼_MAX_RANGE):
        try:
            img: Img = get_img(controller)
        except Exception as e:
            logger.debug(f"获取截图失败: {e}")
            continue

        greem_reco_detail = context.run_recognition(
            "匹配绿色",
            img,
            pipeline_override={
                "匹配绿色": {
                    "recognition": {
                        "type": "ColorMatch",
                        "param": {
                            "roi": 溜鱼_GY_ROI,
                            "lower": [32, 178, 160],
                            "upper": [60, 244, 197],
                        },
                    },
                }
            },
        )
        if greem_reco_detail is None:
            logger.warning(f"识别绿色错误: {greem_reco_detail}")
            return
        gbox = greem_reco_detail.box
        if gbox is None:
            logger.debug("识别不到绿色")
            return

        yellow_reco_detail = context.run_recognition(
            "匹配黄色",
            img,
            pipeline_override={
                "匹配黄色": {
                    "recognition": {
                        "type": "ColorMatch",
                        "param": {
                            "roi": 溜鱼_GY_ROI,
                            "lower": [216, 189, 80],
                            "upper": [253, 252, 186],
                        },
                    },
                }
            },
        )
        if yellow_reco_detail is None:
            logger.warning(f"识别黄色错误: {yellow_reco_detail}")
            return
        ybox = yellow_reco_detail.box
        if ybox is None:
            logger.debug("识别不到黄色")
            return

        green_x = gbox.x + gbox.w / 2
        yellow_x = ybox.x + ybox.w / 2
        diff = green_x - yellow_x

        if (diff_abs := abs(diff)) < option.溜鱼_midpoint_pix_range:
            time.sleep(option.溜鱼_midpoint_sleep_time / 1000)
            continue

        key = Win_virtual_key.A.value.code if diff < 0 else Win_virtual_key.D.value.code
        controller.post_key_down(key).wait()
        time.sleep((diff_abs + max(0, option.溜鱼_midpoint_pix_range / 2 - 1)) / 200)
        controller.post_key_up(key).wait()

    logger.debug(f"溜鱼超出 {溜鱼_MAX_RANGE} 次循环")


def 卖鱼买换饵(context: Context, 买饵次数: int, /):
    logger.info(f"执行 {卖鱼买换饵.__name__}")

    controller = context.tasker.controller

    _get_img = functools.partial(get_img, controller)

    if not reco_钓鱼按钮(context, _get_img()):
        logger.error(f"{卖鱼买换饵.__name__} 钓鱼按钮 识别不到")
        return False

    # region: 市场界面
    for action, sleep_time in (
        (lambda: controller.post_click_key(Win_virtual_key.Q.value.code), 1.5),
        (lambda: controller.post_click(100, 280), 1),  # 归流鱼舱
        (lambda: controller.post_click(710, 645), 1),  # 一键出售
        (lambda: controller.post_click(780, 470), 2),  # 确认
        (lambda: controller.post_click(640, 640), 1),  # 点击空白
        (
            lambda: controller.post_click_key(Win_virtual_key.VK_ESCAPE.value.code),
            1.5,
        ),
    ):
        if context.tasker.stopping:
            logger.info("手动终止")
            raise Manual_stop
        action().wait()
        time.sleep(sleep_time)
    # endregion

    # region: 商店界面
    controller.post_click_key(Win_virtual_key.R.value.code).wait()
    time.sleep(1.5)
    匹配万能鱼饵_reco_detail = context.run_recognition(
        "匹配万能鱼饵",
        _get_img(),
        pipeline_override={
            "匹配万能鱼饵": {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [19, 71, 460, 610],
                    },
                },
            }
        },
    )
    if 匹配万能鱼饵_reco_detail is None:
        logger.error(f"{卖鱼买换饵.__name__} 匹配万能鱼饵 识别错误")
        return False
    if 匹配万能鱼饵_reco_detail.box is None:
        logger.error(f"{卖鱼买换饵.__name__} 匹配万能鱼饵 识别不到")
        return False
    for ocr_res in 匹配万能鱼饵_reco_detail.all_results:
        if not isinstance(ocr_res, OCRResult):
            logger.error("匹配万能鱼饵 OCR 结果不是 OCRResult 类型")
            return False
        if ocr_res.score < 0.9 or ocr_res.text != "万能鱼饵":
            continue

        controller.post_click(ocr_res.box[0], ocr_res.box[1]).wait()
        time.sleep(1)
        匹配万能鱼饵价格_reco_detail = context.run_recognition(
            "匹配万能鱼饵价格",
            _get_img(),
            pipeline_override={
                "匹配万能鱼饵价格": {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": [1162, 577, 31, 31],
                        },
                    },
                }
            },
        )
        if 匹配万能鱼饵价格_reco_detail is None:
            logger.error(f"{卖鱼买换饵.__name__} 匹配万能鱼饵价格 识别错误")
            return False
        if 匹配万能鱼饵价格_reco_detail.box is None:
            logger.error(f"{卖鱼买换饵.__name__} 匹配万能鱼饵价格 识别不到")
            return False
        all_results = 匹配万能鱼饵价格_reco_detail.all_results
        if not type_match(all_results, Sequence[OCRResult]):
            logger.error("匹配万能鱼饵价格 OCR 结果不是 Sequence[OCRResult] 类型")
            return False
        if len(all_results := [r for r in all_results if r.score > 0.8]) != 1:
            logger.warning(f"匹配万能鱼饵价格 匹配数不是 1: {all_results}")
            continue
        if all_results[0].score < 0.96 or all_results[0].text != "5":
            continue

        for _ in range(买饵次数):  # 买 n * 99 个
            for action, sleep_time in (
                (lambda: controller.post_click(1218, 636), 0.5),  # 加满
                (lambda: controller.post_click(1074, 688), 1),  # 购买
                (lambda: controller.post_click(774, 476), 1.5),  # 确认
                (lambda: controller.post_click(640, 540), 1),  # 点击空白
            ):
                if context.tasker.stopping:
                    logger.info("手动终止")
                    raise Manual_stop
                action().wait()
                time.sleep(sleep_time)

        # 获取剩余鱼鳞币
        匹配鱼鳞币_reco_detail = context.run_recognition(
            "匹配鱼鳞币",
            _get_img(),
            pipeline_override={
                "匹配鱼鳞币": {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": [884, 22, 300, 38],
                        },
                    },
                }
            },
        )
        if 匹配鱼鳞币_reco_detail is None:
            logger.error(f"{卖鱼买换饵.__name__} 匹配鱼鳞币 识别错误")
            return False
        all_results = 匹配鱼鳞币_reco_detail.all_results
        if not type_match(all_results, Sequence[OCRResult]):
            logger.error("匹配鱼鳞币 OCR 结果不是 Sequence[OCRResult] 类型")
            return False
        currency_list: list[int] = [
            int(text)
            for r in all_results
            if r.score > 0.9 and (text := r.text.replace(",", "")).isdigit()
        ]
        if 匹配鱼鳞币_reco_detail.box is None:
            logger.error(f"{卖鱼买换饵.__name__} 匹配鱼鳞币 识别不到")
            return False

        if len(currency_list) == 2:
            logger.info(f"鱼鳞币: {currency_list[0]:,}")
            logger.info(f"方斯: {currency_list[1]:,}")
        else:
            logger.warning("识别货币失败")

        controller.post_click_key(  # 退出到初始
            Win_virtual_key.VK_ESCAPE.value.code
        ).wait()
        time.sleep(1)

        break
    # endregion

    # region: 换饵
    for action, sleep_time in (
        (lambda: controller.post_click_key(Win_virtual_key.E.value.code), 1.5),
        (lambda: controller.post_click(496, 360), 1),
        # 点两次，无论如何都会进详细页面
        (lambda: controller.post_click(496, 360), 1),
        (
            lambda: controller.post_click_key(Win_virtual_key.VK_ESCAPE.value.code),
            1,
        ),
        (lambda: controller.post_click(780, 472), 0.5),  # 更换
    ):
        if context.tasker.stopping:
            logger.info("手动终止")
            return False
        action().wait()
        time.sleep(sleep_time)
    # endregion

    return True


FISH_RECO_FIAL_NUM_MAX = 50


@AgentServer.custom_action("fish_fork_2")
class Fish_fork_2(CustomAction):
    @override
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        option: Fish_option = get_option(context, argv)
        logger.debug(f"{option=}")

        def post_click_key(key: Win_virtual_key) -> None:
            controller.post_click_key(key.value.code).wait()

        fail_num: int = 0
        with suppress(Manual_stop):
            while not context.tasker.stopping:
                if (
                    option.终止时间 is not None
                    and option.终止时间 <= datetime.datetime.now().astimezone()
                ):
                    logger.info("到达终止时间")
                    return True

                controller: Controller = context.tasker.controller
                img: Img = get_img(controller)

                if reco_上鱼(context, img):
                    溜鱼(context, option)
                    time.sleep(2)  # 等待弹出获鱼界面
                    controller = context.tasker.controller  # 刷新 controller
                    img = get_img(controller)  # 刷新 img
                    # 继续，直接进入获鱼识别

                if reco_获鱼(context, img):
                    post_click_key(Win_virtual_key.VK_ESCAPE)
                    continue

                if _res := reco_满舱_or_无饵(context, img):
                    if not option.卖鱼买换饵开关:
                        logger.info(f"钓鱼结束: {_res}")
                        return True
                    logger.info(_res)
                    if not 卖鱼买换饵(context, option.买饵次数):
                        logger.error(f"{卖鱼买换饵.__name__} 失败")
                        return False
                    continue

                if reco_钓鱼按钮(context, img):
                    post_click_key(Win_virtual_key.F)
                    time.sleep(0.3)  # 等待弹出溜鱼界面
                    fail_num = 0
                    continue

                if fail_num > FISH_RECO_FIAL_NUM_MAX:
                    logger.error("多次识别失败，终止运行")
                    return False
                logger.debug("什么都没识别到，尝试按 F")
                post_click_key(Win_virtual_key.F)
                fail_num += 1

        return True
