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
degrees = [0,90,180,270]
fums = [1,2,3]
grids = [5]
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
# 划分数据集为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


#tensor to numpy
x_train_tensor = torch.from_numpy(X_train)
x_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

 #gpu environment: transfer into cuda
if torch.cuda.is_available():
    x_train_tensor = x_train_tensor.cuda()
    x_test_tensor = x_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()

#Cominbe dataset
train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)

#Create dataset loader
batch = 64
train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_data_loader = Data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

   
class HeatMapCNN(nn.Module):
    def __init__(self):
        super(HeatMapCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,padding=1)# 8*9*15*15 pading =1 为了利用边缘信息
        self.conv2 = nn.Conv2d(in_channels=9,out_channels=9,kernel_size=3)# 8*9*13*13
        # self.pool1 = nn.AvgPool2d(2) # 8*9*6*6
        # self.conv3 = nn.Conv2d(in_channels=9,out_channels=9,kernel_size=3)# 8*9*4*4
        # self.fc1 = nn.Linear(9 * 4 * 4, 1)#8*144  5mm
        # self.fc1 = nn.Linear(9 * 12 * 12, 1)#8*144  10mm    
        self.fc1 = nn.Linear(9 * 13 * 13, 1)#8*144  10mm    
        # self.fc1 = nn.Linear(9 * 28 * 28, 1)#8*144  10mm    
    def forward(self,x):
        x = fc.relu(self.conv1(x))
        x = fc.relu(self.conv2(x))
        # x = self.pool1(x)
        # x = fc.relu(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        return x

# 创建模型实例
model = HeatMapCNN()
print(model)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000002) #lr
# 记录训练和测试过程中的损失
train_losses = []  # 训练损失
test_losses = []  # 测试损失

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for images, labels in train_data_loader:
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
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# 训练完成后，你可以使用模型进行预测等任务
model.eval()
pre = model(x_test_tensor)
if torch.cuda.is_available():
    pre = pre.gpu()
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

# 绘制训练和测试损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
# plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
