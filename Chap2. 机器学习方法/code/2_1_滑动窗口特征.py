import pandas as pd

# 创建时间序列数据
data = {'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'value': [100, 105, 110, 120, 130, 125, 135, 140, 145, 150]}
df = pd.DataFrame(data)

# 设置日期为索引
df.set_index('date', inplace=True)

# 滑动窗口的均值（窗口大小为3）
df['rolling_mean'] = df['value'].rolling(window=3).mean()

# 滑动窗口的标准差（窗口大小为3）
df['rolling_std'] = df['value'].rolling(window=3).std()

# 滑动窗口的最大值（窗口大小为3）
df['rolling_max'] = df['value'].rolling(window=3).max()

# 滑动窗口的最小值（窗口大小为3）
df['rolling_min'] = df['value'].rolling(window=3).min()

print(df)