import pandas as pd

# 构造时间序列数据
data = {'date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'value': [100, 105, 110, 115, 120]}
df = pd.DataFrame(data)

# 设置日期为索引
df.set_index('date', inplace=True)

# # 创建滞后特征（滞后1期和滞后2期）
# df['lag_1'] = df['value'].shift(1)  # 滞后1期
# df['lag_2'] = df['value'].shift(2)  # 滞后2期


lags = [1, 2, 3]
for lag in lags:
    df[f'lag_{lag}'] = df['value'].shift(lag)


df['day_of_week'] = df.index.dayofweek
df['lag_1']       = df['value'].shift(1)
df['lag_1_dow']   = df['lag_1'] * df['day_of_week']  # 结合星期几和滞后特征

print(df)