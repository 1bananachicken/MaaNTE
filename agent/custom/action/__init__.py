from .AutoFish.auto_fish_new import *
from .AutoFish.auto_fish import *
from .AutoFish.auto_buy_fish_bait import *
from .AutoFish.auto_sell_fish import *
from .auto_make_coffee import *
from .auto_rhythm import *
from .rhythm.feats.select_song import *
from .rhythm.feats.repeat import *
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
    "AutoRhythm",
    "AutoRhythmSelectSong",
    "AutoRhythmRepeat",
    "RealTimeTaskAction",
    "AutoTetris",
]