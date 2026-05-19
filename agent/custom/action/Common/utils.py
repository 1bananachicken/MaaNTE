import time


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img


def click_rect(controller, rect, delay=0.001):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    controller.post_touch_down(cx, cy).wait()
    time.sleep(delay)
    controller.post_touch_up().wait()
