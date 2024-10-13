import pandas as pd

# 创建时间序列数据
# data = {'date': pd.date_range(start='2024-01-01', periods=15, freq='D'),
#         'value': list(range(1, 16))}
data = {'date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'value': [100, 105, 110, 120, 130]}
df = pd.DataFrame(data)

# 设置日期为索引
df.set_index('date', inplace=True)

# 一阶差分（当前值减去前一天的值）
df['diff_1'] = df['value'].diff(periods=1)

# 二阶差分（计算一阶差分的差分）
df['diff_2'] = df['diff_1'].diff(periods=1)

# # 周差分
# df['diff_7'] = df['value'].diff(periods=7)

print(df)