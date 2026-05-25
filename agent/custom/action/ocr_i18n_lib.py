"""前端引述
from ..ocr_i18n_lib import verify_single_task_ocr as ocr_lib

>>feedback ots:
if ocr_lib("呼叫名稱", "前端roi輸出文字"):
    print(" pass ")
例：
if ocr_lib("FishNewGameResult", "閒閉"):
    print("通過！")
"""
# =====================================================================

import re
import difflib

#語言格式模板
"""
 "呼叫名稱": {
    "desc": "輸出文字",
    "expected": [
        "简中判定字A", "简中判定字B", "简中判定字C", 
        "繁中判定字A", "繁中判定字B", "繁中判定字C",
        r"英語判定字A", r"英語判定字B", r"英語判定字C",  # 英=必加 r 與 \s*
         ^敘述： r 鎖定字串原貌，\s* 空格判定
        "日語判定字A", "日語判定字B", "日語判定字C",
        "韓語判定字A", "韓語判定字B", "韓語判定字C"
]
"""

# =====================================================================
#語言任務對照庫
MULTILANG_OCR_LIBRARY = {
    # ClaimRewards
    # 整合以下: ClaimRewardsActivityPoints & MakeCoffeeClaim
    "ClaimClaimButtom": {
        "desc": "领取獎勵",
        "expected": [
            "领取", "領取", r"(?i)CLAIM", r"(?i)Claim",
            "受け取る", "取る", "수령"
        ]
    },
    # 整合以下: ClaimRewardsBattlePassExp & ClaimRewardsBattlePassRewards
    "ClaimRewardsClaimAll": {
        "desc": [
            "领取环期赏令经验", "领取历练奖赏"
        ],
        "expected": [
            "全部领取", "全部領取", "全部", "领取", "領取",
            r"(?i)CLAIM\s*ALL", r"(?i)Claim\s*All",
            "一括受取", "受取", "모두 수령","모두","수령"
        ]
    },
    # FountainCheckin
    # 整合以下: FountainFindButton & FountainCheckinIfExit
    "FountainCheckPoolName": {
        "desc": "识别喷泉打卡按钮",
        "expected": [
            "纳库佩达之池", "納庫佩達之池", "佩达之池", "佩達之池",
            r"(?i)NACUPEDA'S\s*POOL", r"(?i)Nacupeda's\s*Pool", r"(?i)Nacupeda's", r"(?i)Pool",
            "ナクペイダの池", "ナクペイダの池", "나쿠펜다의 연못", "나쿠펜다의"
        ]
    },
    "FountainMakeWish": {
        "desc": "识别许愿选项并按F",
        "expected": [
            "虔诚许愿", "虔誠許願", "许愿", "許願",
            r"(?i)MAKE\s*A\s*SINCERE\s*WITH", r"(?i)Make\s*a\s*Sincere\s*Wish", r"(?i)Make\s*a\s*Sincere",
            "敬虔な願い", "敬虔な願い", "敬虔な願い", "나쿠펜다의 연못", "나쿠펜다의"
        ]
    },
    # 整合 RealTimeSkipStory 系列: RealTimeSkipStoryDialog & 
    "FountainSkipStoryDialog": {
        "desc": "跳过剧情对话框",
        "expected": [
            "是否跳过剧情", "是否跳過劇情","是否跳过", "是否跳過",
            r"(?i)SKIP\s*STORY", r"(?i)Skip\s*Story",
            "ストーリーをスキップしますか", "リーをスキッ", "스토리를 스킵하시겠습니까", "스킵하시겠"
        ]
    },
    # 整合以下: FountainSkipStoryDialogConfirm & 确认收益
    # 整合 FISH 系列: FishNewConfirmBuyBait & FishNewConfirmSellFish
    # 整合 Core2 系列: PinkPawHeist_Once & 
    # 整合 RealTimeSkipStory 系列: RealTimeSkipStoryDialogConfirm & 
    "CheckCheckButton": {
        "desc": "确认按钮",
        "expected": [
            "确认记录", "確認記錄", "記録更新",
            "确认", "確認", "確定", "确定",
            r"(?i)CONFIRN", r"(?i)Confirm", "OK",
            "확인"
        ]
    },
    # MakeCoffee
    "MakeCoffeeStart": {
        "desc": "确认开始营业",
        "expected": [
            "开始营业", "開始營業", "开始", "開始",
            r"(?i)OPEN\s*SHOP", r"(?i)Open\s*Shop",
            "営業開始", "영업 개시"
        ]
    },
    # PinkPawHeist
    "PinkPawHeist_DetectXiaoZhi": {
        "desc": "检测小吱交互按钮",
        "expected": [
            "小支", "小吱", "ちぃちゃん",
            r"(?i)CHIZ", r"(?i)Chiz",
            "치즈"
        ]
    },
    "PinkPawHeist_OCR_Join": {
        "desc": "OCR识别-我要参加",
        "expected": [
            "我加入", "我要参加", "参加する", "参加",
            r"(?i)I'M\s*IN", r"(?i)I'm\s*in",
            "참가하기"
        ]
    },
    "PinkPawHeist_AfterStory": {
        "desc": "在详情页点进入按钮",
        "expected": [
            "进入", "進入", "入る",
            r"(?i)ENTER", r"(?i)Enter",
            "들어가기"
        ]
    },
    "PinkPawHeist_WaitReward": {
        "desc": "等待进地图",
        "expected": [
            "本局收益", "今回の収益", "収益",
            r"(?i)ROUND\s*EARNINGS", r"(?i)Round\s*Earnings",
            "본 라운드 수익","라운드"
        ]
    },
    # WithdrawMoney 收益領取list
    "WithdrawMoneyClickCafe": {
        "desc": "进入一咖舍",
        "expected": [
            "一咖舍", "咖舍", "壱のコーヒー", "壱の", "コーヒー",
            r"(?i)THE\s*CAFE\s*BY\s*ORIGEN", r"(?i)The\s*Cafe\s*by\s*Origen",
            "별미 카페"
        ]
    },
    "WithdrawMoneyWithdraw": {
        "desc": "提取收益",
        "expected": [
            "提取收益", "収益受取", "収益",
            r"(?i)WITHDRAW\s*EARNINGS", r"(?i)Withdraw\s*Earnings",
            "수익 인출", "수익", "인출"
        ]
    },
    # 点击补货 > 改名 > RestockButton
    # 整合以下: 点击补货 & 确认补货
    "RestockButton": {
        "desc": "点击补货",
        "expected": [
            "补货", "補貨", "補充",
            r"(?i)RESTOCK", r"(?i)Restock",
            "재고 보충", "재고", "보충"
        ]
    },
    # 选择24小时 > 改名 > Chouse24Hr
    "Chouse24Hr": {
        "desc": "选择24小时",
        "expected": [
            "24小时", "24小時", "24時間", 
            r"(?i)24\s*H", r"(?i)24\s*h", r"(?i)24H", r"(?i)24h",
            "24시간", "24"
        ]
    },
    # 库存已满 > 改名 > Inventoryfull
    "Inventoryfull": {
        "desc": "库存已满",
        "expected": [
            "库存已满无法存入更多材料", "庫存已滿無法存入更多材料", "库存已满", "庫存已滿",
            r"(?i)INVEBTORY\s*fULL\s*CANNOT\s*STORE\s*MORE\s*MATERIALS",
            r"(?i)Inventory\s*full\s*Cannot\s*store\s*more\s*materials",
            r"(?i)INVEBTORY\s*fULL", r"(?i)Inventory\s*full",
            "空きがありません", "空きが", "ありません",
            "재고가 가득 차서", "재고가", "가득", "차서"
        ]
    },
    # 送货上门 > 改名 > HomeDelivery
    "HomeDelivery": {
        "desc": "送货上门",
        "expected": [
            "送货上门", "送貨上門",
            r"(?i)DELIVERY", r"(?i)Delivery",
            "引き渡し", "配達", "交付", "宅配",
            "배송하기"
        ]
    },
    # Fish
    # 整合以下: FishUseBait & FishNewUseBait 
    "FishUseBait": {
        "desc": "点击更换使用万能饵, If ok = 仍有饵，完成后开始钓鱼",
        "expected": [
            "更换", "更換", "替换", "替換", "変更", "變更", "切换", "切換",
            "入れ替え", "切り替え", "切替",
            r"(?i)SWITCH", r"(?i)Switch",r"(?i)SWAP", r"(?i)Swap",
            "교체"
        ]
    },
    # 整合+改名以下: FishGotoBuyBait & FishBaitExit & FishNewBaitExit
    #   > FishNewGotoBuyBait & FishNewStoreBuyBait & FishNewConfirmBuyBait 
    "FishGlobalBuyButton": {
        "desc": "if万能饵=0 >点击购买按钮 >渔具商店 轉交 auto_buy_fish_bait 执行实际购买",
        "expected": [
            "购买", "購買", "購入",
            r"(?i)PURCHASE", r"(?i)Purchase",
            "구매"
        ]
    },
    # FishNew
    "FishNewBaitLack": {
        "desc": "缺少鱼饵",
        "expected": [
            "需要装备鱼饵才可以钓鱼", "需要裝備魚餌才可以釣魚", "需要装备鱼饵", "需要裝備魚餌",
            r"(?i)EQUIP\s*BAIT\s*BEFORE\s*FISHING", r"(?i)Equip\s*bait\s*before\s*fishing",
            r"(?i)EQUIP\s*BAIT",r"(?i)BEFORE\s*FISHING", r"(?i)Equip\s*bait", r"(?i)before\s*fishing",
            "釣りをするには釣り餌を装着する必要があります", "釣り餌を装着する必要",
            "미끼를 장착해야 낚시할 수 있습니다","미끼를","장착해야","낚시할","있습니다"
        ]
    },
    "FishNewFishFull": {
        "desc": "渔获已满",
        "expected": [
            "渔获已满", "漁獲已滿", "渔获", "已滿",
            r"(?i)YOUR\s*FISH\s*HOLD\s*IS\s*FULL", r"(?i)Your\s*fish\s*hold\s*is\s*full",
            r"(?i)YOUR\s*FISH", r"(?i)IS\s*FULL", r"(?i)Your\s*fish", r"(?i)is\s*full",
            "生け簀が一杯です 魚を売却後に再度お試しください","生け簀が一杯です", "魚を売却後に",
            "저장고가 가득 찼습니다 일부 어획물을 판매한 뒤 다시 시도해 주세요"
        ]
    },
    "FishNewBaitScrollMouse": {
        "desc": "滚动+更换 鱼饵界面",
        "expected": [
            "更换鱼饵", "更換魚餌", "更换", "更換", "鱼饵", "魚餌",
            r"(?i)SWITCH\s*BAIT", r"(?i)Switch\s*Bait",
            "釣り餌交換", "交換",
            "미끼 교체", "미끼", "교체"
        ]
    },
    # 整合以下: FishNewOnStore & FishNewStoreScrollMouse & FishNewExitStore
    #   > FishNewExitStoreSellFish
    "FishNewOnStore": {
        "desc": "到达商店界面",
        "expected": [
            "渔具商店", "漁具商店", "商店", "釣具店", "商店", "店",
            r"(?i)FISHING\s*TACKLE\s*SHOP", r"(?i)Fishing\s*Tackle\s*Shop",
            "낚시용품 상점", "낚시용품", "상점"
        ]
    },
    # FishScene
    "FishSceneOnFishPrepare": {
        "desc": "判断正在钓鱼准备界面，继续流程",
        "expected": [
            "钓鱼准备", "釣魚準備", "釣り準備", "准备", "準備", "釣",
            r"(?i)FISHING\s*PREPARATIONS", r"(?i)Fishing\s*Preparations",
            "낚시 준비", "낚시", "준비"
        ]
    },
    "FishScenePrepareStartFishButton": {
        "desc": "钓鱼准备界面>开始钓鱼按钮",
        "expected": [
            "开始钓鱼", "開始釣魚", "釣り開始", "開始", "开始", "釣",
            r"(?i)START\s*FISHING", r"(?i)Start\s*Fishing",
            "낚시 시작", "낚시", "시작"
        ]
    },
    "FishSceneInWorldNextToFish": {
        "desc": "判断正在大世界钓鱼台边，继续业务流程",
        "expected": [
            "钓鱼", "釣魚", "釣り",
            r"(?i)FISHING", r"(?i)Fishing",
            "낚시"
        ]
    },
    # Status
    "InEscMenu": {
        "desc": "在Esc菜单内的猎人等级",
        "expected": [
            "猎人等级", "獵人等級", "人等",
            r"(?i)HUNTER\s*LEVEL", r"(?i)Hunter\s*Level",
            "ハンターレベル", "ハンター", "レベル",
            "헌터 레벨", "헌터", "레벨"
        ]
    },
    "InExplorationGuideMenu": {
        "desc": "在探索指南菜单内",
        "expected": [
            "探索指南", "探索", "指南",
            r"(?i)EXPLORATION\s*GUIDE", r"(?i)Exploration\s*Guide",
            "探検ガイド", "探検", "ガイド",
            "탐색 안내서", "탐색", "안내서"
        ]
    },
    "InBattlePassMenu": {
        "desc": "在环期赏令菜单内-P1",
        "expected": [
            "历练奖赏", "歷練獎賞", "奖赏", "獎賞",
            r"(?i)EXPERIENCE\s*REWARDS", r"(?i)Experience\s*Rewards",
            "鍛錬の証", "鍛錬", "証",
            "수련 보상", "수련", "보상"
        ]
    },
    "InScarboroughFairMenu": {
        "desc": "在斯卡布罗集市内",
        "expected": [
            "弧盘研募", "弧盤研募", "弧盤研究プロジェクト", "弧盤", "研募",
            r"(?i)ARC\s*RESEARCH", r"(?i)Arc\s*Research",
            "아크 연구 모집", "아크", "연구", "모집"
        ]
    },
    "InEventsMenu": {
        "desc": "在活动菜单内",
        "expected": [
            "活动", "活動", "活",
            r"(?i)EVENT", r"(?i)Event",
            "イベント", "이벤트"
        ]
    },
    "InBagMenu": {
        "desc": "在背包菜单内",
        "expected": [
            "背包",
            r"(?i)INVENTORY", r"(?i)Inventory",
            "バッグ", "가방"
        ]
    },
    "InCharactersMenu": {
        "desc": "在角色菜单内",
        "expected": [
            "个人信息", "個人資訊", "信息", "資訊",
            r"(?i)PROFILE", r"(?i)Profile" r"(?i)INFO", r"(?i)Info",
            "メッセージ", "情報", "정보"
        ]
    },
    "InCityTycoonMenu": {
        "desc": "在都市大亨菜单内",
        "expected": [
            "都市大亨", "都市", "大亨", "大",
            r"(?i)CITY\s*TYCOON", r"(?i)City\s*Tycoon",
            "シティ名声", "シティ", "名声",
            "도시 타이쿤", "도시", "타이쿤"
        ]
    },
    # Core2
    # 整合以下: PinkPawHeist_CheckGate2Once & PinkPawHeist_CheckGateOnce
    "PinkPawHeist_CheckGate2Once": {
        "desc": "单次检测铁门-強制",
        "expected": [
            "强制开启", "強制開啟", "強制解錠", "強制",
            r"(?i)FORCE\s*OPEN", r"(?i)Force\s*Open",
            "강제 오픈", "강제", "오픈"
        ]
    },
    "PinkPawHeist_CheckDoorOnce": {
        "desc": "单次检测开门",
        "expected": [
            "开门", "開門",
            r"(?i)OPEN", r"(?i)Open",
            "ドアを開ける", "開",
            "문 열기", "문", "열기"
        ]
    },
    "PinkPawHeist_CheckReward": {
        "desc": "本局收益-识别於地图上",
        "expected": [
            "本局收益", "今回の収益", "収益", "收益",
            r"(?i)ROUND\s*EARNINGS", r"(?i)Round\s*Earnings",  #1
            r"(?i)THIS\s*ROUND\s*PROFIT", r"(?i)This\s*Round\s*Profit",  #2
            "이 라운드 수익", "이", "라운드", "수익"
        ]
    },
    # Tools
    # 整合以下: PinkPawHeist_EvacuateOnce &
    "PinkPawHeist_CheckEvacuateOnce": {
        "desc": "检测-撤离确认按钮",
        "expected": [
            "确认撤离", "確認撤離", "確認撤离", "撤离", "確認",
            r"(?i)CONFIRM\s*RETREAT", r"(?i)Confirm\s*Retreat",
            "撤退確定", "撤退", "確定",
            "철수 확인", "철수", "확인"
        ]
    },
    # RealTimeTeleport
    "RealTimeTeleportWitte": {
        "desc": "传送至维特海默塔",
        "expected": [
            "维特海默塔", "維特海默塔", "特海默塔",
            r"(?i)WERTHEIMER\s*TOWER", r"(?i)Wertheimer\s*Tower",
            "ヴェルテハイモー塔", "ヴェルテ", "ハイモー塔",
            "베르트하이머 탑", "베르트하이머"
        ]
    },
    # 整合以下: RealTimeConfirmTeleportWitte & RealTimeConfirmTeleportPhone
    "RealTimeConfirmTeleportWitte": {
        "desc": "确认传送",
        "expected": [
            "传送", "傳送", "転送", "送",
            r"(?i)TELEPORT", r"(?i)Teleport",
            "텔레포트"
        ]
    },
    "RealTimeTeleportPhone": {
        "desc": "传送至电话亭",
        "expected": [
            "传送点", "傳送點",
            r"(?i)WERTHEIMER\s*TOWER", r"(?i)WERTHEIMER\s*TOWER",
            "転送ポイント", "転送", "ポイント", "送",
            "텔레포트 포인트", "텔레포트", "포인트"
        ]
    },
    # RhythmScene
    "RhythmSceneOnResults": {
        "desc": "判断正在结算界面",
        "expected": [
            "得分",
            r"(?i)SCORE", r"(?i)Score",
            "スコア", "점수"
        ]
    },
    "RhythmSceneStartButton": {
        "desc": "选歌界面-开始演奏按钮",
        "expected": [
            "开始演奏", "開始演奏", "演奏を開始する", "演奏", "開始", "开始",
            r"(?i)PLAY", r"(?i)Play",
            "연주 시작", "연주", "시작"
        ]
    },
    "RhythmSceneFindSong": {
        "desc": "选歌界面-在歌曲列表中匹配指定歌曲",
        "expected": [
            "迷星叫", "迷星", "星叫", "迷", "星", "叫",
            r"(?i)MAYOIUTA", r"(?i)Mayoiuta"
        ]
    },
    # SceneLogin
    "__ScenePrivateLoginContinue": {
        "desc": "等待登录页面按钮 >继续",
        "expected": [
            "游戏", "遊戲",
            r"(?i)GAME", r"(?i)Game",
            "ゲーム", "게임"
        ]
    },
    "__ScenePrivateCloseLoginAnnouncement": {
        "desc": "关闭登录公告界面",
        "expected": [
            "公告",
            r"(?i)NOTICE", r"(?i)Notice",
            "お知らせ", "공지"
        ]
    },
    "__ScenePrivateLoginMonthlyPassConfirm": {
        "desc": "领取月卡，无法跳过-必须领取",
        "expected": [
            # A = 异晶开采凭证
            "异晶开采凭证", "異晶開採憑證", "異晶採掘免許証", "異晶", "异晶", 
            r"(?i)RIFTCRYSTAL\s*MINING\s*PERMIT", r"(?i)Riftcrystal\s*Mining\s*Permit",
            "이상 수정 채굴증", "이상", "수정", "채굴증",
            # B = 剩余
            "剩余", "剩餘", r"(?i)REMAINING", r"(?i)Remaining", #1
            r"(?i)RIFTCRYSTAL\s*MINING\s*PERMIT", r"(?i)Riftcrystal\s*Mining\s*Permit", #2
            "残り", "잔여",
            # C = 收集奖励
            "收集奖励", "收集獎勵", "收集", "報酬",
            r"(?i)CLAIM", r"(?i)Claim",            
            "수집 보상", "수집", "보상",
            # D = 剩余天数
            "剩余天数", "剩餘天數", "残り日数", "剩", "天", "数",
            r"(?i)REMAINING:",  r"(?i)Remaining:",
            "남은 일수", "남은", "일수",
            # E = 今日奖励
            "按下领取今日奖励", "按下領取今日獎勵", "今日奖励", "今日獎勵", "今日",          
            r"(?i)CLAIM\s*DAILY\s*REWARD", r"(?i)Claim\s*Daily\s*Reward",
            "クリックして本日の報酬をお受け取りください", "本日の報酬", "本日", "報酬",
            "오늘 보상 수령", "오늘", "보상", "수령"
        ]
    },
    # TetrisScene
    "TetrisSceneExit": {
        "desc": "泯除方块-检测退出按钮",
        "expected": [
            "退出", "終了",
            r"(?i)EXIT",r"(?i)Exit", r"(?i)QUIT",r"(?i)Quit",
            "종료"
        ]
    },
    "TetrisSceneGameIdle": {
        "desc": "泯除方块-检测游戏场景-标记（游戏空闲）",
        "expected": [
            r"(?i)TETROMINOES", r"(?i)Tetrominoes"
        ]
    },
    "TetrisSceneLoading": {
        "desc": "泯除方块 - 检测加载界面",
        "expected": [
            #// @i18n-skip
            "%"
        ]
    },
    "TetrisSceneWorldPrompt": {
        "desc": "泯除方块-检测大世界入口提示",
        "expected": [
            "泯除方块", "泯除方塊", "泯除方",
            r"(?i)TETROMINOES", r"(?i)Tetrominoes",
            "排除パズル", "排除", "パズル",
            "테트로미노", "테트", "로미노"
        ]
    },
    "TetrisScenePrepareOneSingle": {
        "desc": "泯除方块-检测单人模式入口",
        "expected": [
            "单人游戏", "單人遊戲", "人",
            r"(?i)SOLO", r"(?i)Solo",
            "シングルプレイ", "シングル", "プレイ",
            "싱글플레이", "싱글플", "레이"
        ]
    },
    "TetrisScenePrepareOneMulti": {
        "desc": "泯除方块-检测多人模式入口",
        "expected": [
            "多人游戏", "多人遊戲", "多人",
            r"(?i)MULTIPLAYER", r"(?i)Multiplayer", #1
            r"(?i)MULTI", r"(?i)Multi", #2
            "マルチプレイ", "マルチ", "プレイ",
            "멀티플레이 게임", "멀티플레이", "게임"
        ]
    },
    "TetrisSceneStartMatch": {
        "desc": "泯除方块-检测开始匹配按钮",
        "expected": [
            "开始匹配", "開始匹配", "マッチング開始", "匹配", "開始",
            r"(?i)START\s*MATCHMAKING", r"(?i)Start\s*Matchmaking",
            "매칭 시작", "매칭", "시작"
        ]
    },
    "TetrisSceneMatching": {
        "desc": "泯除方块-检测匹配中界面",
        "expected": [
            "匹配进行中", "匹配進行中", "匹配", "行中",
            r"(?i)MATCHMAKING", r"(?i)Matchmaking",
            "マッチング中", "マッチング", "中",
            "매칭 중", "매칭", "중"
        ]
    },
    "TetrisSceneMatchend": {
        "desc": "泯除方块-检测对局结束界面",
        "expected": [
            r"(?i)MATCHEND", r"(?i)Matchend"
        ]
    },
    #
    ##standby 預備用
    #
    "Ptory": {
        "desc": "在钮",
        "expected": [
            "进入", "進入", "入る"
            r"(?i)ENTER", r"(?i)Enter",
            "들"
        ]
    }  ##,
}

# =====================================================================
# 2. 核心邏輯比對：被動驗證（由前端指定的單一 Task 驅動，一命中立刻 return 中斷）
def verify_single_task_ocr(task_name, frontend_ocr_text, threshold=0.5):
    """
    被動式對照：前端執行特定節點時呼叫此函數。本函數只驗證指定的 task_name 清單，不跟進其他任務。
    
    :param task_name: 前端執行當前任務節點名稱 (例: 'FishNewGameResult')
    :param frontend_ocr_text: 前端切好 ROI 並做完 OCR 後傳進來的原始純文字 (可能殘缺、變形或有錯字)
    :param threshold: 錯字模糊比對門檻 (0.5 表一半字元正確即放行)
    :return: True (對照成功，前端可執行 Action) / False (不匹配，跳過)
    """
    # 如前端傳入一庫裡未設定任務，直接回傳不匹配
    if task_name not in MULTILANG_OCR_LIBRARY:
        return False
        
    task_data = MULTILANG_OCR_LIBRARY[task_name]
    cleaned_text = frontend_ocr_text.strip()
    
    # -------------------------------------------------------------
    # 判定壹：精準字型判定 (處理文字前後殘缺、空格、單字命中)
    for pattern in task_data["expected"]:
        # re.search 配合庫裡的 r"" 與 \s* 能夠完美相容各式各樣的殘缺英文句
        if re.search(pattern, cleaned_text):
            # 中斷點 A：只要精準命中，立刻回傳 True 並中斷函數
            return True
            
    # -------------------------------------------------------------
    # 判定貳：精準字型判定失效時啟動，對應 OCR 經典錯字
    for pattern in task_data["expected"]:
        # 排除帶有正則符號的英文規則 (因為英文有 \s* 符號會干擾純字元比對)
        if "(" in pattern or "\\" in pattern:
            continue
            
        # 使用 Python 內建機制計算前端錯字與庫內標準字的相似度 (0.0 ~ 1.0)
        similarity_score = difflib.SequenceMatcher(None, cleaned_text, pattern).ratio()
        
        # 只要相似度大於等於設定的門檻 (例如 0.5)
        if similarity_score >= threshold:
            # 中斷點 B：模糊匹配錯字成功，立刻回傳 True 並中斷函數
            return True
            
    # 如果精準和模糊都沒對，回傳 False
    return False
