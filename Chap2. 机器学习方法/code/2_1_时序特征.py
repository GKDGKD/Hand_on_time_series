import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 生成模拟时间序列数据
np.random.seed(42)
data = pd.DataFrame({
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'value': np.random.randint(50, 150, size=100)
})
data.set_index('date', inplace=True)

# 1. 构造滞后特征
for lag in range(1, 4):  # 构造1到3天的滞后特征
    data[f'lag_{lag}'] = data['value'].shift(lag)

# 2. 构造滑动窗口统计特征
data['rolling_mean_3'] = data['value'].rolling(window=3).mean()
data['rolling_std_3'] = data['value'].rolling(window=3).std()

# 3. 构造时间特征
data['day_of_week'] = data.index.dayofweek
data['day_of_month'] = data.index.day
data['month'] = data.index.month

# 4. 去掉缺失值（由于滞后和滚动窗口导致的缺失）
data.dropna(inplace=True)

# 5. 训练集和测试集划分
X = data.drop(columns=['value'])  # 特征
y = data['value']  # 目标
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 保持时间顺序

# 6. 训练机器学习模型（随机森林回归）
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 模型预测
y_pred = model.predict(X_test)

# 8. 评估模型表现
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 展示部分预测结果
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())