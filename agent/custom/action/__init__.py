from .AutoFish.auto_fish_new import *
from .AutoFish.auto_fish import *
from .AutoFish.auto_buy_fish_bait import *
from .AutoFish.auto_sell_fish import *
from .auto_make_coffee import *
from .rhythm.feats.play import *
from .rhythm.feats.repeat_decision import *
from .rhythm.feats.select_song import *
from .Common.click import *
from .realtime_task import *
from .auto_tetris import *


__all__ = [
    "AutoFishNew",
    "AutoMakeCoffee",
    "AutoFish",
    "AutoBuyFishBait",
    "AutoSellFish",
    "ClickOverride",
    "AutoTetris",
    "AutoRhythmPlay",
    "AutoRhythmRepeatDecision",
    "AutoRhythmSelectSong",
    "RealTimeTaskAction",
]