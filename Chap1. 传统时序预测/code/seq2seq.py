import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 生成时间序列数据
np.random.seed(42)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.normal(size=len(t))

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

# 划分数据集
def create_sequences(data, input_length, output_length):
    sequences = []
    labels = []
    for i in range(len(data) - input_length - output_length):
        sequences.append(data[i:i+input_length])
        labels.append(data[i+input_length:i+input_length+output_length])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

input_length = 20  # 输入长度
output_length = 10  # 输出预测的步长
X, y = create_sequences(data_scaled, input_length, output_length)

# 数据分为训练和测试集
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 定义Encoder-Decoder模型
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell  # 返回LSTM的隐状态和细胞状态

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x, hidden, cell):
        # x shape: (1, batch_size, output_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)  # 通过全连接层获取最终预测
        return prediction, hidden, cell

# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq_len):
        # input_seq shape: (batch_size, seq_len, input_size)
        batch_size = input_seq.shape[0]
        input_seq = input_seq.permute(1, 0, 2)  # 转换为(seq_len, batch_size, input_size)
        
        # 编码阶段
        hidden, cell = self.encoder(input_seq)

        # 初始化解码阶段输入
        decoder_input = torch.zeros(1, batch_size, 1).to(input_seq.device)  # 预测未来的起始值
        
        predictions = []
        for t in range(target_seq_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions.append(prediction.squeeze(0))  # 去掉 seq_len 维度
            decoder_input = prediction  # 下一步的输入是当前的预测值

        predictions = torch.stack(predictions, dim=1)  # (batch_size, seq_len, output_size)
        return predictions

# 设置模型参数
input_size = 1  # 单变量时间序列
hidden_size = 64  # LSTM 隐藏层大小
output_size = 1  # 单步输出

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
model = Seq2Seq(encoder, decoder)

# 训练模型
def train(model, X_train, y_train, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        output = model(X_train.unsqueeze(-1), output_length)
        loss = criterion(output.squeeze(-1), y_train)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 将训练集输入到模型中训练
train(model, X_train, y_train, epochs=100)

# 测试模型
model.eval()
with torch.no_grad():
    test_predictions = model(X_test.unsqueeze(-1), output_length)

# 可视化结果
test_predictions = test_predictions.squeeze(-1).cpu().numpy()
y_test = y_test.cpu().numpy()

plt.plot(np.arange(len(y_test)), y_test[:, 0], label='Truth')
plt.plot(np.arange(len(test_predictions)), test_predictions[:, 0], label='Prediction')
plt.legend()
plt.show()
