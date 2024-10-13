import pandas as pd
import chinese_calendar as cc  # 导入chinese-calendar库

# 创建一个包含时间戳的数据框
data = {'timestamp': pd.date_range(start='2024-10-01', periods=15, freq='D')}
df = pd.DataFrame(data)

# 定义一个函数，判断某一天是否是中国的假期
def is_holiday(date):
    return cc.is_holiday(date)  # 返回True或False

# 定义一个函数，判断某一天是否是调休日
def is_workday(date):
    return cc.is_workday(date)

# 提取时间特征
df['year']        = df['timestamp'].dt.year          # 提取年份
df['month']       = df['timestamp'].dt.month        # 提取月份
df['day']         = df['timestamp'].dt.day            # 提取日期
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 提取星期几 (0 = 周一, 6 = 周日)
df['is_weekend']  = df['day_of_week'].isin([5, 6]) # 是否是周末
df['hour']        = df['timestamp'].dt.hour          # 提取小时
# df['is_holiday']  = df['timestamp'].isin(['2023-01-01'])  # 根据假日列表添加特征

# 节假日特征
df['is_holiday'] = df['timestamp'].apply(is_holiday)  # 判断是否为假期
df['is_workday'] = df['timestamp'].apply(is_workday)  # 判断是否为调休日

print(df)
