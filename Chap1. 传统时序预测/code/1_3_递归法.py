import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成一个简单的时间序列数据
np.random.seed(42)
n = 20  # 数据长度
data = np.array([i + np.random.normal(0, 1) for i in range(n)])

# 定义滞后特征的构造函数
def create_lagged_features(data, lag=1):
    df = pd.DataFrame(data, columns=['value'])
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['value'].shift(i)
    return df.dropna()

# 构造滞后特征（假设我们使用1个滞后特征）
lag = 1
train_data = create_lagged_features(data, lag=lag)

# 分离特征和标签
X_train = train_data.drop('value', axis=1).values  # 滞后特征作为输入
y_train = train_data['value'].values  # 当前值作为输出

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 递归预测未来的5个时间步
steps = 5
history = data.tolist()
predictions = []

for t in range(steps):
    # 使用最新的历史数据来构造滞后特征
    latest_data = np.array(history[-lag:]).reshape(1, -1)  # 使用最近的 lag 个值作为输入
    
    # 预测下一个时间步
    next_pred = model.predict(latest_data)[0]
    predictions.append(next_pred)
    
    # 将预测值加入到历史数据中
    history.append(next_pred)

# 打印预测结果
print(f"递归法预测未来 {steps} 步的值：{predictions}")

# 可视化结果
plt.plot(np.arange(len(data)), data, label="history")
plt.plot(np.arange(len(data), len(data) + steps), predictions, label="predictions", color="orange")
plt.legend()
plt.show()
