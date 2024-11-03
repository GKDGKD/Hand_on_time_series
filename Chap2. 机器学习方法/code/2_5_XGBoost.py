import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor, plot_importance, plot_tree

# 生成模拟的时序数据
np.random.seed(42)
date_range = pd.date_range(start='2024-01-01', periods=200, freq='D')
data = pd.DataFrame({'value': np.sin(np.arange(200)/10) + np.random.normal(0, 0.1, 200)}, index=date_range)

# 构造滞后特征和滚动统计特征
def create_features(df, target_col, lags, window):
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
    df.dropna(inplace=True)
    return df

# 构建特征
lags = 3  # 滞后3步
window = 5  # 滑动窗口大小为5
data = create_features(data, 'value', lags, window)

# 构建训练集和测试集
X = data.drop(columns=['value'])
y = data['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# ## 方式1：原生接口，转换为DMatrix格式
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)

# # 参数设置
# params = {
#     'objective'    : 'reg:squarederror', # 回归目标
#     'max_depth'    : 3,                  # 树的最大深度
#     'learning_rate': 0.1,                # 学习率
#     'eval_metric'  : 'rmse'             # 评估指标
# }

# # 训练模型
# evals = [(dtrain, 'train'), (dtest, 'eval')]
# model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=False)

# # 预测与评估
# y_pred = model.predict(dtest)


## 方式2：sklearn接口，XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

# 特征重要性
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance (Original XGBoost)")
# plt.show()
plt.savefig("XGBoost_Feature_Importance.png")

# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("XGBoost Predictions vs Actual (Original Interface)")
# plt.show()
plt.savefig("XGBoost_Predictions_vs_Actual.png")


# 绘制决策树
xgb.plot_tree(model, num_trees=0, rankdir="LR")
plt.title("XGBoost Decision Tree (Original Interface)")
# plt.show()
plt.savefig("XGBoost_Decision_Tree.png")
