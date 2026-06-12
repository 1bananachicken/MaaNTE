import time
from ..Common.utils import get_image, click_rect_multiple



def make_croissant(context):
    """制作牛角包"""
    context.run_action("MakeCoffeeLiteDish1_SelectIngredient1")
    context.run_action("MakeCoffeeLiteDish1_SelectIngredient2")
    context.run_action("MakeCoffeeLiteDish1_Confirm")


def make_cake(context):
    """制作小蛋糕"""
    context.run_action("MakeCoffeeLiteDish2_SelectIngredient1")
    context.run_action("MakeCoffeeLiteDish2_SelectIngredient2")
    context.run_action("MakeCoffeeLiteDish2_Confirm")


def make_bread(context):
    """制作面包"""
    context.run_action("MakeCoffeeLiteDish3_SelectIngredient1")
    context.run_action("MakeCoffeeLiteDish3_SelectIngredient2")
    context.run_action("MakeCoffeeLiteDish3_Confirm")


def wait_and_claim(context, controller, check_freq=0.5):
    """等待并点击领取按钮"""
    while True:
        if context.tasker.stopping:
            return False
        img = get_image(controller)
        claim_result = context.run_recognition("MakeCoffeeClaim", img)
        if claim_result and claim_result.hit:
            click_rect_multiple(
                controller,
                [
                    claim_result.box.x,
                    claim_result.box.y,
                    claim_result.box.w,
                    claim_result.box.h,
                ],
            )
            time.sleep(1)
            return True
        time.sleep(check_freq)


def press_key_f(controller):
    """按下并释放 F 键"""
    KEY_F = 70
    controller.post_key_down(KEY_F)
    time.sleep(0.1)
    controller.post_key_up(KEY_F)
