from datetime import datetime, timedelta
import pytz


def ms_timestamp_diff_to_dhm(timestamp1_ms, timestamp2_ms):
    """
    将两个毫秒级时间戳的差值转换为天-时-分格式

    参数:
        timestamp1_ms (int): 第一个毫秒时间戳
        timestamp2_ms (int): 第二个毫秒时间戳

    返回:
        str: 格式化为"X天-X时-X分"的字符串
    """
    # 计算时间戳之间的绝对差值（毫秒）
    diff_ms = abs(timestamp2_ms - timestamp1_ms)

    # 转换为秒
    diff_seconds = diff_ms / 1000

    # 计算天、小时、分钟
    days = int(diff_seconds // (24 * 3600))
    remaining_seconds = diff_seconds % (24 * 3600)
    hours = int(remaining_seconds // 3600)
    remaining_seconds %= 3600
    minutes = int(remaining_seconds // 60)

    # 返回中文格式的结果
    return f"{days}天-{hours}时-{minutes}分"


def is_current_period(timestamp_ms, timezone="Asia/Shanghai"):
    """
    判断毫秒级时间戳是否在当前周和当前月

    参数:
        timestamp_ms: 毫秒级时间戳
        timezone: 时区字符串，默认为"Asia/Shanghai"（北京时间）

    返回:
        tuple: (is_current_week, is_current_month)
    """

    tz = pytz.timezone(timezone)
    timestamp_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0, tz)
    now = datetime.now(tz)

    # 计算当前周的开始（本周或上周一05:00:00）
    # Python中，weekday()返回0-6，0是周一，6是周日
    days_since_monday = now.weekday()  # 距离最近过去的周一的天数

    # 计算到本周一的天数
    week_start = now.replace(hour=5, minute=0, second=0, microsecond=0) - timedelta(
        days=days_since_monday
    )

    # 如果当前时间早于周一5点，则使用上周一作为周期开始
    if now.weekday() == 0 and now.hour < 5:  # 如果是周一且不到5点
        week_start = week_start - timedelta(days=7)  # 使用上周一

    # 计算下周一05:00:00作为本周结束
    week_end = week_start + timedelta(days=7)

    # 计算当前月的开始（当月1号05:00:00）
    if now.day == 1 and now.hour < 5:
        # 如果是1号但不到5点，使用上个月1号作为开始
        if now.month == 1:
            month_start = datetime(now.year - 1, 12, 1, 5, 0, 0, 0, tzinfo=tz)
        else:
            month_start = datetime(now.year, now.month - 1, 1, 5, 0, 0, 0, tzinfo=tz)
    else:
        # 否则使用本月1号
        month_start = now.replace(day=1, hour=5, minute=0, second=0, microsecond=0)
        # 如果已经过了1号5点，但当前日期小于1号，则需要往前调整一个月
        if now.day < 1 or (now.day == 1 and now.hour < 5):
            if month_start.month == 1:
                month_start = month_start.replace(year=month_start.year - 1, month=12)
            else:
                month_start = month_start.replace(month=month_start.month - 1)

    # 计算下个月1号05:00:00作为当月结束
    if month_start.month == 12:
        month_end = datetime(month_start.year + 1, 1, 1, 5, 0, 0, 0, tzinfo=tz)
    else:
        month_end = datetime(
            month_start.year, month_start.month + 1, 1, 5, 0, 0, 0, tzinfo=tz
        )

    # 判断是否在当前周和当前月
    is_current_week = week_start <= timestamp_datetime < week_end
    is_current_month = month_start <= timestamp_datetime < month_end

    return is_current_week, is_current_month