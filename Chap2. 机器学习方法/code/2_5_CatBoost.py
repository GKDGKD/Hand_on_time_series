import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 生成模拟的时序数据
np.random.seed(42)
date_range         = pd.date_range(start='2024-01-01', periods=200, freq='D')
data               = pd.DataFrame({'value': np.sin(np.arange(200)/10) + np.random.normal(0, 0.1, 200)}, index=date_range)
# data               = data.reset_index()
# data['month']      = data['index'].dt.month
# data['is_weekend'] = data['index'].dt.dayofweek.isin([5, 6])
print(data.head())

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


# CatBoost模型训练（sklearn接口）
model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, verbose=0)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

# 特征重要性
feature_importances = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('CatBoost Feature Importance')
# plt.show()
plt.savefig("CatBoost_Feature_Importance.png")


# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("CatBoost Predictions vs Actual")
# plt.show()
plt.savefig("CatBoost_Predictions_vs_Actual.png")
