import numpy as np
import math
import torch
import torch.nn as nn
from torch import optim
import matplotlib as plt
from time  import time
import torch.nn.functional as F
import matplotlib.pyplot  as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch.utils.data as Data

def create_dataset(X, y):
    features = []
    targets = []
    
    for i in range(0, len(X)): 
        data = [[i] for i in X[i]] # 序列数据  
        label = [y[i]] # 标签数据
        
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    
    # 返回
    return np.array(features,dtype=np.float32), np.array(targets,dtype=np.float32)


# split data
# x_train, x_test, y_train, y_test

def split_dataset(x, y, train_ratio=0.8):

    x_len = len(x) # 特征数据集X的样本数量
    train_data_len = int(x_len * train_ratio) # 训练集的样本数量
    
    x_train = x[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集
    
    x_test = x[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集
    
    # 返回值
    return x_train, x_test, y_train, y_test


# set seed
seed = 2
batch = 8

# set which file to use to build model and what is the grid size
filenum = 3
gsize = 30 #5,10,15,20,30,40,50
overlapping_step = 3 # 1,3,5
shuffle = True

dataset_x = []
dataset_y = []
import random


#DIF实验室路径：'/home/durrr/phd/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_5.txt' 
#MBP 路径:     '/Users/darren/资料/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_5.txt'
with open('/Users/darren/资料/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_{overlapping_step}.txt'.format(size = gsize, fnum = filenum,overlapping_step = overlapping_step), 'r') as f:
# with open('/home/durrr/phd/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping{overlapping_step}.txt'.format(size = gsize, fnum = filenum,overlapping_step = overlapping_step), 'r') as f:
    lines = f.readlines()
    if shuffle:
      random.Random(seed).shuffle(lines)
    else:
      pass
    # print(lines[10])
    for line in lines:
        line = line.strip("\n")
        dataset_x.append(line.split("|")[0].split(","))
        dataset_y.append(line.split("|")[1])
        

# print(len(dataset_x))

dataset_x
lable = [float(y) for y in dataset_y]
input_x = []
for grp in dataset_x:
  input_x.append([float(z) for z in grp])


input_x,lable = create_dataset(input_x, lable)
x_train, x_test, y_train, y_test = split_dataset(input_x, lable, train_ratio=0.80)

nsample,nx,ny = x_train.shape
x_train_2d = x_train.reshape(nsample, nx*ny)

nsamplet,nxt,nyt = x_test.shape
x_test_2d = x_test.reshape(nsamplet, nxt*nyt)

#tensor to numpy
x_train_tensor = torch.from_numpy(x_train_2d)
x_test_tensor = torch.from_numpy(x_test_2d)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

#gpu environment: transfer into cuda
if torch.cuda.is_available:
    x_train_tensor = x_train_tensor.cuda()
    x_test_tensor = x_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()

#Combine data
train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor, y_test_tensor)


# print(x_train)
#GRU with Attention model

# set paramemaers
# GRU
gru_units = 128
num_layer = 1
acti = 'relu'
drop = 0
regu = None #regularizers.l2(1e-4)
batch = 8
#16！

# optimizer
lr = 0.001
mom = 0.01

#complie
callback_file = None
epo = 1000


#Create dataloader
loader = Data.DataLoader(dataset = train_dataset, batch_size = batch, shuffle = shuffle)


class Attention(nn.Module):
   def __init__(self, hidden_size):
       super(Attention, self).__init__()
       self.hidden_size = hidden_size
       self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
       self.v = nn.Linear(hidden_size, 1, bias=False)

   def forward(self, hidden, encoder_outputs):
       max_len = encoder_outputs.size(1)
       repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
       energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
       attention_scores = self.v(energy).squeeze(2)
       attention_weights = nn.functional.softmax(attention_scores, dim=1)
       context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
       return context_vector, attention_weights

class GRUwithAttention(nn.Module):
   def __init__(self,input_size,hidden_size,dropout = 0.5):
      super(GRUwithAttention,self).__init__()
      self.gru = nn.GRU(input_size,hidden_size,batch_first = True) #  data length ->128
      self.attention = Attention(hidden_size)
      self.fc = nn.Linear(hidden_size,256) #128 -> 256
      self.fc1 = nn.Linear(256,64) #256->64
      self.fc2 = nn.Linear(64,1) #64->1
      self.dropout = nn.Dropout(drop)

   def forward(self,input,hidden):
      output, hidden = self.gru(input,hidden) 
      output ,attention_weight = self.attention(hidden[-1],output)
      output = torch.relu(output)
      out = self.dropout(out)
      output = self.fc(output)
      output = torch.relu(output)
      output = self.fc1(output)
      output = torch.relu(output)
      output = self.fc2(output)
      return output,hidden
   
      
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUwithAttention(x_train_tensor.shape[1],gru_units).to(device=device)
print(model)

criterion = nn.L1Loss()

optimize = optim.SGD(model.parameters(),lr =lr,momentum=mom,nesterov = False)
time0 = time()

hx  = x_train_tensor
cx  = torch.randn((x_train_tensor.shape[0],x_train_tensor.shape[1]))

maes = []
mses = []
rmses = []
r2s = []

#metrics
def metrics(predict,expected):
    if torch.cuda.is_available():
        predict = predict.cpu()
    predict = predict.detach().numpy()
    expected = expected.cpu()

    mae = mean_absolute_error(expected,predict)
    mse = mean_squared_error(expected,predict)
    rmse = mean_squared_error(expected,predict,squared=False)
    r2=r2_score(expected,predict)

    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    r2s.append(r2)
   
total_losss=[]
for e in range(epo):
   total_loss = 0
   for step, (batch_x,batch_y) in enumerate(loader):
    optimize.zero_grad()
    hx,cx = model(batch_x,None)
    
    metrics(hx,batch_y)
    loss= criterion(hx,batch_y)
    loss.backward()

    optimize.step()
    total_loss = total_loss + loss.item()
    if step == len(loader) -1:
       total_losss.append(total_loss)
    print(f"Epoch: {e+1}, Step: {step}, Loss:{loss.item()}")

print("\nTraining Time(in minutes) = ",(time()-time0)/60)
   

for e in range(epo):
   optimize.zero_grad()
   hx,cx = model(x_train_tensor,None)
  
   metrics(hx,y_train_tensor)
   loss= criterion(hx,y_train_tensor)
  #  mses.append(loss.item())
   loss.backward()

   optimize.step()
   print(f"Epoch {e+1}, Loss:{loss.item()}")

print("\nTraining Time(in minutes) = ",(time()-time0)/60)

# #evaluation model

model.eval()
pre,_ = model(x_test_tensor,None)
if torch.cuda.is_available():
    pre = pre.cpu()
pre = pre.detach().numpy()
y_test_tensor = y_test_tensor.cpu()

mae = mean_absolute_error(y_test_tensor,pre)
mse = mean_squared_error(y_test_tensor,pre)
rmse = mean_squared_error(y_test_tensor,pre,squared=False)
r2=r2_score(y_test_tensor,pre)

print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)


epos = np.arange(epo)+1
# mae_plt = plt.plot(epos,maes,label='MAE')
# mse_plt = plt.plot(epos,mses,label = 'MSE')
# rmse_plt = plt.plot(epos,rmses,label = 'RMSE')
# r2_plt = plt.plot(epos,r2s,label='R2')
loss_plt = plt.plot(epos,total_losss,label = "Total_loss")
plt.title('GRUWithAttention/Metrics_outfile{fnum}/gridized{size}mm'.format(size = gsize, fnum = filenum))
plt.xlabel("Epo")
plt.ylabel("Metrics Value")
plt.legend()
plt.show()
# plt.savefig("GRU_Metrics_outfile{filenum}_gridized{size}mm")