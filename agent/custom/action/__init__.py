from .AutoFish.auto_fish import *
from .AutoFish.auto_buy_fish_bait import *
from .AutoFish.auto_sell_fish import *
from .auto_make_coffee import *
from .rhythm.feats.play import *
from .rhythm.feats.repeat_decision import *
from .rhythm.feats.select_song import *
from .Common.click import *
from .realtime_task import *
from .Navi import *
from .pinkpaw.pinkpaw_core1 import *
from .pinkpaw.pinkpaw_core2 import *
from .pinkpaw.pinkpaw_reward_logger import *
from .auto_tetris import *
from .AutoFish.auto_fish_withoutCV import *
from .SoundTrigger.SoundDodgeAction import *
from .auto_f_scroll import *
from .Movement.mouse_move import *
from .Movement.character_move import *
from .Common.alt_click import *
from .furniture_claim import *
from .auto_piano.action import *
from .withdraw_money_choose_item import *
from .DatasetCollection.autonomous_driving_dataset_recorder import *

__all__ = [
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
    "NaviWebSocketAction",
    "PinkPawHeistScheme1Action",
    "PinkPawHeistScheme2Action",
    "PinkPawRewardSummary",
    "AutoFishWithoutCV",
    "SoundDodgeAction",
    "AutoFScroll",
    "AltClick",
    "FurnitureClaim",
    "AutoPlayPiano",
    "WithdrawMoneyChooseItem",
    "AutonomousDrivingDatasetRecorder",
]
