import numpy as np
import torch
import os
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [10]

class HeatMapCNN(nn.Module):
    def __init__(self):
        super(HeatMapCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,padding=1)# 8*9*15*15 pading =1 为了利用边缘信息
        self.conv2 = nn.Conv2d(in_channels=9,out_channels=9,kernel_size=3)# 8*9*13*13  
        self.fc1 = nn.Linear(9 * (3*grids[0]-2) * (3*grids[0]-2), 1)  
    def forward(self,x):
        x = fc.relu(self.conv1(x))
        x = fc.relu(self.conv2(x))
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        return x

def read_data(f_num,d,degree):
    if degree == 0:
        # 设置图像文件夹和标签文件夹的路径
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/labels'
    else:
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/rotate/{degree}/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/rotate/{degree}/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # 创建空的训练数据列表，用于存储图像和标签
    X = []  # 用于存储图像数据
    y = []  # 用于存储标签

    # 遍历图像文件列表
    for image_path in image_files:
        # 获取图像文件名，不包括路径和文件扩展名
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 构建相应的标签文件路径
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # 读取标签文件内容
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()  # 假设标签是一行文本

        # 打开图像文件并进行预处理
        with Image.open(image_path) as img:
            # 这里可以添加图像预处理步骤，例如将图像调整为固定大小、归一化等
            # img = img.convert("L")
            img = np.array(img)  # 将图像转化为NumPy数组
        img = img.transpose((2, 0, 1))
        # img = img.reshape(1,15,15)
        # 将图像数据和标签添加到列表
        X.append(img)
        y.append(label)
    return X,y
  

# f_num = 3
# d = 5
# X,y = read_data(3,5)
X = []  # 用于存储图像数据
y = []  # 用于存储标签

for fum in fums:
    for grid in grids:
        for degree in degrees:
            X_fum_grid_degree,y_fum_grid_degree = read_data(fum,grid,degree)
            X+=X_fum_grid_degree
            y+=y_fum_grid_degree

# 将X和y转化为NumPy数组
X = np.array(X)
y = np.array(y)


# 归一化图像数据
X = X / 255.0  # 假设使用0-255的像素值
batch = 8

# 划分数据集为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# #tensor to numpy
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

#  #gpu environment: transfer into cuda
if torch.cuda.is_available():
    X_train_tensor = X_train_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()

# #Cominbe dataset
# train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
# val_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)

# #Create dataset loader

# train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
# val_data_loader = Data.DataLoader(val_dataset, batch_size=batch, shuffle=False)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    # # 获取训练集和测试集
    # X_train, X_test = X_train_tensor[train_idx], X_train_tensor[test_idx]
    # y_train, y_test = y_train_tensor[train_idx], y_train_tensor[test_idx]

    # 将剩下的9份按照8:2的比例划分为训练集和验证集
    train_size = int(0.8 * len(train_idx))
    X_train_fold, X_val_fold = X_train[:train_size], X_train[train_size:]
    y_train_fold, y_val_fold = y_train[:train_size], y_train[train_size:]

    # X_test = X_test.astype('float32')
    # y_test = y_test.astype('float32')
    X_train_fold = X_train_fold.astype('float32')
    X_val_fold = X_val_fold.astype('float32')
    y_train_fold = y_train_fold.astype('float32')
    y_val_fold = y_val_fold.astype('float32')


    #tensor to numpy
    X_train_tensor_fold = torch.from_numpy(X_train_fold)
    X_val_tensor_fold = torch.from_numpy(X_val_fold)
    y_train_tensor_fold = torch.from_numpy(y_train_fold)
    y_val_tensor_fold = torch.from_numpy(y_val_fold)
    # X_test_tensor = torch.from_numpy(X_test)
    # y_test_tensor = torch.from_numpy(y_test)

    #gpu environment: transfer into cuda
    if torch.cuda.is_available():
        X_train_tensor_fold = X_train_tensor_fold.cuda()
        X_val_tensor_fold = X_val_tensor_fold.cuda()
        # X_test_tensor = X_test_tensor.cuda()
        # y_test_tensor = y_test_tensor.cuda()
        y_train_tensor_fold = y_train_tensor_fold.cuda()
        y_val_tensor_fold = y_val_tensor_fold.cuda()
        

    #Cominbe dataset
    train_dataset_fold = Data.TensorDataset(X_train_tensor_fold,y_train_tensor_fold)
    # test_dataset = Data.TensorDataset(X_test_tensor,y_test_tensor)
    val_dataset_fold = Data.TensorDataset(X_val_tensor_fold,y_val_tensor_fold)

    # #Create dataset loader
    train_data_loader_fold = Data.DataLoader(train_dataset_fold, batch_size=batch, shuffle=True)
    # test_data_loader = Data.DataLoader(test_dataset, batch_size=batch, shuffle=True)
    val_data_loader_fold = Data.DataLoader(val_dataset_fold, batch_size=batch, shuffle=False)

    # 在这里你可以使用 X_train, y_train 进行模型的训练
    # 使用 X_val, y_val 进行模型的验证
    # 创建模型实例
    model = HeatMapCNN()
    # print(model)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000002) #lr
    # 记录训练和测试过程中的损失
    train_losses = []  # 训练损失
    test_losses = []  # 测试损失

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        for images, labels in train_data_loader_fold:
            optimizer.zero_grad()
            outputs = model(images)
            al_label = labels.unsqueeze(1)
            loss = criterion(outputs, al_label)
            loss.backward()
            optimizer.step()
            # 查看梯度情况
            # for name, parms in model.named_parameters(): 
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
        train_losses.append(loss.item() )
        for images, labels in val_data_loader_fold:
            outputs = model(images)
            loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # 使用 X_test, y_test 进行最终的测试评估
    # 训练完成后，你可以使用模型进行预测等任务
    model.eval()
    pre = model(X_val_tensor_fold)
    if torch.cuda.is_available():
        pre = pre.gpu()
    pre = pre.detach().numpy()
    y_val_tensor_fold = y_val_tensor_fold.cpu()

    mae = mean_absolute_error(y_val_tensor_fold,pre)
    mse = mean_squared_error(y_val_tensor_fold,pre)
    rmse = mean_squared_error(y_val_tensor_fold,pre,squared=False)
    r2=r2_score(y_val_tensor_fold,pre)

    #记录的是 测试集的结果
    # result_file_path = f'/Users/darren/资料/SPIF_DU/Croppings/result/{grids[0]}mm/tencrosvalidationresult_validate.txt'
    # with open(result_file_path,"a") as file:
    #         file.write(str(fold+1))
    #         file.write("\nMAE"+str(mae))
    #         file.write("\nMSE"+str(mse))
    #         file.write("\nRMSE"+str(rmse))
    #         file.write("\nR2"+str(r2)+"\n")
    # plt.figure()
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # # plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # # 打印当前折数和各集合的大小
    # print(f"Fold {fold + 1} - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
   
#validation的结果
model.eval()
pre = model(X_test_tensor)
if torch.cuda.is_available():
    pre = pre.gpu()
pre = pre.detach().numpy()
y_test_tensor = y_test_tensor.cpu()

mae = mean_absolute_error(y_test_tensor,pre)
mse = mean_squared_error(y_test_tensor,pre)
rmse = mean_squared_error(y_test_tensor,pre,squared=False)
r2=r2_score(y_test_tensor,pre)

print("Fold:",fold+1)
print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)