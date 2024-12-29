import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from 指数平滑预测 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取Excel文件中的Sheet6
file_path = '交通数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet6')
df = pd.read_excel('交通数据.xlsx', sheet_name='Sheet6')
history_data = df['日交通量'].tolist()
pred_data,once_data,second_data,third_data=pre_next(history_data, alpha=0.5)

# 选择所需的数据
data = third_data

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

#转为张量
data_tensor = torch.FloatTensor(scaled_data).view(-1).to(device)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

seq_length = 7 #序列长度
inout_seq= create_inout_sequences(data_tensor, seq_length)

train_size = int(len(inout_seq) * 0.8)
train_inout_seq = inout_seq[:train_size]
test_inout_seq = inout_seq[train_size:]

num_layers=1
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1,num_layers=num_layers):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
model = LSTM()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

epochs = 350
loss_list = []
for i in range(epochs):
    once_train_inout_seq = random.sample(train_inout_seq, 50)
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    loss_list.append(float(f'{single_loss.item():10.8f}'))
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')


model.eval()
pred_test_all = []

for seq, labels in inout_seq:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
        out = model(seq)
        pred_test_all.append(scaler.inverse_transform(out.detach().cpu().numpy().reshape(-1, 1))[0][0])
pred_test_index = [351-i for i in range(len(pred_test_all))]
pred_test_index = pred_test_index[::-1]

pred10 = []
pred10_index = [i+len(scaled_data) for i in range(10)]

for i in range(10):
    in_data = scaled_data[-seq_length:]
    in_data = torch.FloatTensor(in_data).view(-1).to(device)
    model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
    out = model(in_data)
    scaled_data = np.append(scaled_data, out.detach().cpu().numpy().reshape(1,1), axis=0)
    pred = scaler.inverse_transform(out.detach().cpu().numpy().reshape(-1, 1))[0][0]
    pred10.append(pred)

plt.ylim(0,25000)
plt.xlim(0,370)
plt.plot(df['日交通量'],label = '日交通量')
plt.plot(third_data, label='三次指数平滑')
plt.plot(pred_test_index,pred_test_all,label='lstm预测值')
plt.plot(pred10_index, pred10,label='未来10年lstm预测')
plt.legend()
plt.savefig('lstm日交通量copy.png',dpi=300)
plt.cla()
plt.plot(loss_list)
plt.savefig('lstm日交通量losscopy.png', dpi=300)
