import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib as plt
from time  import time
import torch.nn.functional as F
import matplotlib.pyplot  as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
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

# set which file to use to build model and what is the grid size
filenum = 3
gsize = 50 #5,10,15,20,30,40,50
shuffle = True

dataset_x = []
dataset_y = []
import random
# /home/durrr/phd/SPIF_DU/MainFolder/50mm_file/outfile3/trainingfile_50mm_overlapping_5.txt
with open('/home/durrr/phd/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_3.txt'.format(size = gsize, fnum = filenum), 'r') as f:
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

#gpu environment: transfer int cuda
if torch.cuda.is_available():
    x_train_tensor = x_train_tensor.cuda()
    x_test_tensor = x_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()


#Combine data
train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)

# set paramemaers
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
loader = Data.DataLoader(dataset=train_dataset,batch_size=batch,shuffle=shuffle)

class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size,gru_units):
        super(GRUModel,self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,gru_units)
        self.fc = nn.Linear(hidden_size,256)
        self.fc1 = nn.Linear(256,64)
        self.fc2 = nn.Linear(64,1)

    
    def forward(self, input,hidden):
        output,hidden = self.gru(input,hidden)
        output =torch.relu(output)
        output = self.fc(output)
        output =torch.relu(output)
        output = self.fc1(output)
        output =torch.relu(output)
        output = self.fc2(output)
        return output,hidden


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUModel(x_train_tensor.shape[1],gru_units,gru_units).to(device=device)
print(model)


criterion = nn.L1Loss()

optimize = optim.SGD(model.parameters(),lr =lr,momentum=mom)
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
   

for e in range(epo):
   for step, (batch_x,batch_y) in enumerate(loader):
    optimize.zero_grad()
    hx,cx = model(batch_x,None)
    
    metrics(hx,batch_y)
    loss= criterion(hx,batch_y)
    loss.backward()

    optimize.step()
    print(f"Epoch: {e+1}, Step: {step+1}, Loss:{loss.item()}")

print("\nTraining Time(in minutes) = ",(time()-time0)/60)


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
mae_plt = plt.plot(epos,maes,label='MAE')
mse_plt = plt.plot(epos,mses,label = 'MSE')
rmse_plt = plt.plot(epos,rmses,label = 'RMSE')
r2_plt = plt.plot(epos,r2s,label='R2')
plt.title("Metrics_outfile{filenum}/gridized{size}mm")
plt.xlabel("Epo")
plt.ylabel("Metrics Value")
plt.legend()
plt.show()
plt.savefig("GRU_Metrics_outfile{filenum}_gridized{size}mm")

