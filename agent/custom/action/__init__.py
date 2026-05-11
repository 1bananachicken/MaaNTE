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
from .AutoFish.auto_fish_withoutCV import *
from .SoundTrigger.SoundDodgeAction import *
from .auto_f_scroll import *
from .Movement.mouse_move import *
from .Movement.character_move import *
from .Common.alt_click import *
from .furniture_claim import *
from .auto_piano.action import *

__all__ = [
    "AutoFishNew",
    "AutoMakeCoffee",
    "AutoFish",
    "AutoBuyFishBait",
    "AutoSellFish",
    "ClickOverride",
    "AutoRhythmPlay",
    "AutoRhythmRepeatDecision",
    "AutoRhythmSelectSong",
    "RealTimeTaskAction",
    "AutoFishWithoutCV",
    "SoundDodgeAction",
    "AutoFScroll",
    "AltClick",
    "FurnitureClaim",
    "AutoPlayPiano",
]
