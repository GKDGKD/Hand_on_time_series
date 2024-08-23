动手学时间序列预测

目录：

一、传统时序预测
- 1.1 时间序列介绍
- 1.2 平稳性与非平稳性
- 1.3 单元时序与多元时序、单步向前和多步向前
- 1.4 统计方法（均值预测、权重预测等）
- 1.5 ARIMA（由浅入深介绍MA、AR、ARMA、ARIMA、SARIMA）
- 1.6 指数平滑

二、机器学习方法
- 2.1 滞后特征构建
- 2.2 线性回归
- 2.3 SVM
- 2.4 决策树和随机森林
- 2.5 梯度提升树（至少包含XGBoost、LightGBM、CatBoost）

三、深度学习方法
- 3.1 数据集构建
- 3.2 RNN、LSTM、GRU
- 3.3 TCN
- 3.4 Transformer（LogTrans、Reformer、Informer、Autoformer等，可拆分成几节写）
- 3.5 ...


四、前沿模型（进阶）

- 4.1 线性模型前沿研究
- 4.2 状态空间模型前沿研究
- 4.3 生成式模型前沿研究
- 4.4 Transformer类模型前沿研究
- 4.5 大模型前沿研究

五、时序技巧
- 5.1 预测范式选择
- 5.2 特殊预处理技术
- 5.3 傅里叶变换与维纳-辛钦定理
- 5.4 经验模态分解
- 5.5 外生变量使用策略
- 5.6 集成学习策略


六、竞赛实战
- 6.1 AI夏令营-电力需求预测赛
- 6.2 AI夏令营-地球科学

规范：
1. 本课程前3章面向小白，尽量简洁易懂，生动形象，需要原理、代码。后面几章为进阶，可以严谨点。模型原理尽量配图，必须有代码。图片统一放在每一章节的images下，数据集统一放在datasets目录下。
2. 可以使用GPT辅助，但需要校验确保GPT生成的内容正确无误
3. 第2，3章需要介绍模型和编写代码，统一采用96的历史时间窗口，以及96的预测步数。机器学习模型可以调包，深度学习模型基于pytorch构建
4. 第4章面向进阶选手，可以介绍多种比较好的模型，不限于sota模型，也可以介绍一些比较经典、表现好的模型
5. 第5章主要介绍一些时序上常见的技巧，包括傅里叶变换、经验模态分解等，有好的idea欢迎拓展
6. 第6章可讲竞赛或行业经验分享，竞赛可来源于kaggle、讯飞等比赛，一个比赛一个文件夹
7. 目录大纲可根据实际情况需要做微调修改

分工与ddl
整体计划：10.1前完成初版


参考资料：
【THUML】SOTA模型汇总
A Survey on Diffusion Models for Time Series and Spatio-Temporal Data
https://otexts.com/fppcn/
https://github.com/zhouhaoyi/Informer2020

数据集：
https://github.com/zhouhaoyi/ETDataset
https://github.com/laiguokun/multivariate-time-series-data
https://github.com/ServiceNow/N-BEATS 

竞赛：
AI夏令营-电力需求预测赛学习者手册（看看能不能作为时序项目案例）
链接：AI夏令营-电力需求预测赛学习者手册

任务1（简单）
时间序列预测方法：均值预测
链接： 从零入门机器学习竞赛

任务2（中等）
时间序列预测方法：机器学习模型lightgbm
链接：Task2：入门lightgbm，开始特征工程

任务3（较难）
时间序列预测方法：深度学习
链接：Task3：尝试使用深度学习方案
