import numpy as np
import torch
import os
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

def read_data(f_num,d):

    # 设置图像文件夹和标签文件夹的路径
    image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/images'
    label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/labels'

    # 获取图像文件夹中的所有图像文件
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
            img = np.array(img)  # 将图像转化为NumPy数组
        img = img.transpose((2, 0, 1))
        # 将图像数据和标签添加到列表
        X.append(img)
        y.append(label)
    return X,y
  

# f_num = 3
# d = 5
X1,y1 = read_data(1,5)
X2,y2 = read_data(2,5)
X3,y3 = read_data(3,5)
X = X1 + X2 + X3
y = y1 + y2 + y3

# 将X和y转化为NumPy数组
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)
# 归一化图像数据（根据需要）
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
batch = 32
train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_data_loader = Data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

   
class HeatMapCNN(nn.Module):
    def __init__(self):
        super(HeatMapCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,4,3)
        self.fc1 = nn.Linear(4 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.shape[0],-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型实例
model = HeatMapCNN()

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
# 记录训练和测试过程中的性能指标
train_losses = []  # 训练损失
test_accuracies = []  # 测试准确度

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 训练完成后，你可以使用模型进行预测等任务
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')

# 绘制训练和测试的对比折线图
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss', color='blue')
# plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Performance')
plt.legend()
plt.title('Training vs. Testing Performance')
plt.grid(True)
plt.show()