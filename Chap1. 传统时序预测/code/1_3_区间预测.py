import numpy as np
import scipy.stats as st

# # 假设一个正态分布的概率预测
# mean = 2000  # 预测均值
# std_dev = 50  # 标准差

# # 计算特定区间的概率
# prob = st.norm.cdf(2100, loc=mean, scale=std_dev) - st.norm.cdf(1900, loc=mean, scale=std_dev)
# print(f"电力负荷在1900 MW到2100 MW之间的概率为：{prob:.2%}")


from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 示例数据
data = [100, 105, 110, 115, 120]  # 历史数据

# 拟合ARIMA模型 (p, d, q)
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# 进行区间预测（置信水平为95%），预测未来1步
forecast_obj = model_fit.get_forecast(steps=1)
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)

# 输出结果
print(f"预测值：{forecast[0]}")
print(f"95%置信区间：{conf_int[0][0]} - {conf_int[0][1]}")
