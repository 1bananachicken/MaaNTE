from utils.logger import logger


def check_resolution(controller):
    controller.post_screencap().wait()
    w, h = controller.resolution
    if (w, h) == (0, 0):
        return

    if (w, h) == (1280, 720):
        logger.info(f"当前窗口分辨率: {w}x{h} [正常]")
    else:
        logger.warning(
            f"当前窗口分辨率 {w}x{h}，请使用 1280x720 分辨率。"
            "请将游戏设置为 1280x720 窗口化模式，否则部分功能可能异常。"
        )
