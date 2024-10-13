import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 假设有过去10年的人口增长数据
data = [1000, 1050, 1100, 1200, 1250, 1300, 1400, 1450, 1500, 1600]

# 构建 Holt-Winters 指数平滑模型
model = ExponentialSmoothing(data, trend="add", seasonal=None, seasonal_periods=None)
model_fit = model.fit()

# 预测未来5年的人口增长
forecast = model_fit.forecast(steps=5)

# 可视化预测结果
plt.plot(range(10), data, label='Historical Data')
plt.plot(range(10, 15), forecast, label='Forecast')
plt.legend()
plt.show()