import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

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
lags   = 3  # 滞后3步
window = 5  # 滑动窗口大小为5
data   = create_features(data, 'value', lags, window)

# 分割数据
X = data.drop(columns=['value'])
y = data['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 从训练集划分出验证集（例如，20%的数据用于验证）
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)


## 方式1：使用LightGBM的原生接口
# 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)
test_data  = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'objective'    : 'regression',
    'metric'       : 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves'   : 31,
    'learning_rate': 0.05,
    'verbose'      : -1
}

# 训练模型
model = lgb.train(
    params,
    train_data,
    num_boost_round       = 100,
    valid_sets            = [train_data, val_data], # 使用训练集和验证集
)

# 预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)


# ## 方式2：使用sklearn接口
# # 初始化和训练模型
# model = LGBMRegressor(
#     objective='regression',
#     metric='rmse',
#     boosting_type='gbdt',
#     num_leaves=31,
#     learning_rate=0.05,
#     n_estimators=100
# )

# model.fit(X_train, y_train)

# # 预测
# y_pred = model.predict(X_test)


# 评估模型
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

# 绘制特征重要性
lgb.plot_importance(model, max_num_features=10, importance_type='split', figsize=(8, 6))
plt.title("Feature Importance (Original API)")
# plt.show()
plt.savefig("LightGBM_Feature_Importance.png")

# 绘制决策树
lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
plt.title("Tree Structure of the First Boosted Tree")
# plt.show()
plt.savefig("LightGBM_Decision_Tree.png")


# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("LightGBM Predictions vs Actual (Original Interface)")
# plt.show()
plt.savefig("LightGBM_Predictions_vs_Actual.png")
