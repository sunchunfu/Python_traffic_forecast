import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
plt.rcParams['font.sans-serif'] = ['SimHei']  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取Excel文件中的Sheet6
file_path = '交通数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet6')

# 选择所需的数据
data = df['日交通量'].tolist()

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data).reshape(-1,1))

#转为张量
data_tensor = torch.FloatTensor(scaled_data).view(-1).to(device)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw+1):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

seq_length = 5 #序列长度
inout_seq= create_inout_sequences(data_tensor, seq_length)

train_size = int(len(inout_seq) * 0.8)
train_inout_seq = inout_seq[:train_size]
test_inout_seq = inout_seq[train_size:]

class NET(nn.Module):
    def __init__(self, input_size=seq_length,  output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        #self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()


    def forward(self, input_seq):
        x = self.relu(self.fc1(input_seq))
        #x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

model = NET()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

epochs = 500
loss_list = []
for i in range(epochs):
    once_train_inout_seq = random.sample(train_inout_seq, 50)
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    loss_list.append(float(f'{single_loss.item():10.8f}'))
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

model.eval()
model.to(device)
pred_all = []

for seq, labels in inout_seq:
    with torch.no_grad():
        out = model(seq)
        pred_all.append(scaler.inverse_transform(out.detach().cpu().numpy().reshape(-1, 1))[0][0])
pred_all_index = [i+seq_length-1 for i in range(len(pred_all))]

pred_next = []
pred_next_index = [i+len(scaled_data) for i in range(10)]
for i in range(10):
    in_data = scaled_data[-seq_length:]
    in_data = torch.FloatTensor(in_data).view(-1).to(device)
    out = model(in_data)
    scaled_data = np.append(scaled_data, out.detach().cpu().numpy().reshape(1,1), axis=0)
    pred = scaler.inverse_transform(out.detach().cpu().numpy().reshape(-1, 1))[0][0]
    pred_next.append(pred)

plt.ylim(0,25000)
plt.xlim(0,360)
plt.plot(df['日交通量'],label = '日交通量')
plt.plot(pred_all_index,pred_all,label = 'bp预测值')
plt.plot(pred_next_index, pred_next,label = 'bp未来10天预测值')
plt.legend()
plt.savefig('bp日交通量.png', dpi=300)

plt.cla()
plt.plot(loss_list,label = '均方误差损失')
plt.savefig('bp日交通量loss.png', dpi=300)