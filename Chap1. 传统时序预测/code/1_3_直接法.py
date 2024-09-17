import numpy as np
from sklearn.linear_model import LinearRegression

"""
多步向前预测——直接法
为未来每一个时间点创建一个预测模型，分别进行预测。

"""

# 假设我们有过去5天的温度数据
X_train = np.array([[30], [32], [34], [33], [31]])  # 历史温度
y_train_day_1 = np.array([32, 34, 33, 31, 30])  # 第一天预测
y_train_day_2 = np.array([33, 32, 31, 30, 29])  # 第二天预测
y_train_day_3 = np.array([34, 33, 32, 31, 30])  # 第三天预测

# 分别训练3个模型
model_day_1 = LinearRegression().fit(X_train, y_train_day_1)
model_day_2 = LinearRegression().fit(X_train, y_train_day_2)
model_day_3 = LinearRegression().fit(X_train, y_train_day_3)

# 预测未来 3 天
day_1_pred = model_day_1.predict([[31]])  # 第一天预测
day_2_pred = model_day_2.predict([[31]])  # 第二天预测
day_3_pred = model_day_3.predict([[31]])  # 第三天预测

print(day_1_pred, day_2_pred, day_3_pred)


# # 方式二：采用sklearn的MultiOutputRegressor接口实现直接法
# from sklearn.multioutput import MultiOutputRegressor

# # 多输出回归模型
# X_train = np.array([[30], [32], [34], [33], [31]])  # 历史温度
# y_train = np.array([[32, 33, 34], [34, 32, 33], [33, 31, 32], [31, 30, 31], [30, 29, 30]])  # 3天预测

# # 训练一个模型来同时预测3个步长
# multi_target_model = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)

# # 预测未来3天的温度
# forecast = multi_target_model.predict([[31]])
# print(forecast)