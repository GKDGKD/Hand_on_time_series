import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 生成简单时间序列数据
np.random.seed(42)
data = np.array([i + np.random.normal(0, 1) for i in range(50)])

# 滞后特征构造函数
def create_lagged_features(data, lag=1):
    df = pd.DataFrame(data, columns=['value'])
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['value'].shift(i)
    return df.dropna()

# 构造滞后特征（使用滞后1步）
lag = 1
train_data = create_lagged_features(data, lag=lag)

# 特征和标签
X_train = train_data.drop('value', axis=1).values  # 滞后特征
y_train = train_data['value'].values  # 当前值

# 训练随机森林模型（用于直接法和递归法）
model_direct = RandomForestRegressor(n_estimators=300)
model_direct.fit(X_train, y_train)

# 预测步长设置
steps = 10
split = 5  # 前5步用直接法，后5步用递归法
history = data.tolist()
predictions = []

# **直接法**：分别预测前5个时间步的值
for i in range(1, split+1):
    # 构造滞后特征
    lagged_data = np.array(history[-lag:]).reshape(1, -1)
    
    # 用模型预测每个时间步的值
    pred = model_direct.predict(lagged_data)[0]
    predictions.append(pred)
    
    # 将预测值加入历史数据中
    history.append(pred)

# **递归法**：基于前面的预测结果，逐步预测后5个时间步的值
for i in range(split, steps):
    # 构造滞后特征（使用前面的预测值）
    lagged_data = np.array(history[-lag:]).reshape(1, -1)
    
    # 预测下一个时间步
    pred = model_direct.predict(lagged_data)[0]
    predictions.append(pred)
    
    # 将预测值加入历史数据中
    history.append(pred)

# 输出预测结果
print(f"联合法预测未来 {steps} 步的值：{predictions}")

# 可视化结果
plt.plot(np.arange(len(data)), data, label="history")
plt.plot(np.arange(len(data), len(data) + steps), predictions, label="prediction", color="orange")
plt.legend()
plt.show()
