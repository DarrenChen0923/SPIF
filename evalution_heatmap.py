from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
import os
from PIL import Image
import torch.nn as nn
import numpy as np
import torch.nn.functional as fc


## Parameters 
version = 2
grids = [5]
num_epochs = 1000
batch = 64
learning_rate = 0.001
model_path = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/models/model_version{version}_grid{grids[0]}_epo{num_epochs}_batch{batch}_lr{learning_rate}.pth'
degrees = [0,90,180,270]
fums = [1,2,3]

##Read test dataset
def read_data(train_or_test):
        # 设置图像文件夹和标签文件夹的路径
        # set Image file and label file path
    image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/images'
    label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # 创建空的训练数据列表，用于存储图像和标签
    # Create empty list to store iamges and lables
    X = []  # 用于存储图像数据 store images
    y = []  # 用于存储标签 store lables

    # 遍历图像文件列表
    # Iterate Images
    for image_path in image_files:
        # 获取图像文件名，不包括路径和文件扩展名
        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 构建相应的标签文件路径
        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # 读取标签文件内容
        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()  # 假设标签是一行文本

        # 打开图像文件并进行预处理
        # open image file and do preprocessing
        with Image.open(image_path) as img:
            # 这里可以添加图像预处理步骤，例如将图像调整为固定大小、归一化等
            # img = img.convert("L")
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))
        # img = img.reshape(1,15,15)
        # 将图像数据和标签添加到列表
        # put image and label into list
        X.append(img)
        y.append(label)
    return X,y

for fum in fums:
    for grid in grids:
        for degree in degrees:
            X_test,y_test = read_data("test")
           
# 将X和y转化为NumPy数组
# transfer X and y into numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)


# 归一化图像数据
# Normalise
X_test = X_test/255.0 # 使用0-255的像素值

def normalize_mean_std(image):
    mean = np.mean(image)
    stddev = np.std(image)
    normalized_image = (image - mean) / stddev
    return normalized_image

X_test = normalize_mean_std(X_test)


X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


# #tensor to numpy
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

#  #gpu environment: transfer into cuda
if torch.cuda.is_available():
    X_test_tensor = X_test_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()


##定义模型
class HeatMapCNN(nn.Module):
    def __init__(self):
        super(HeatMapCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,padding=1)# 8*9*15*15 pading =1 为了利用边缘信息
        self.conv2 = nn.Conv2d(in_channels=9,out_channels=9,kernel_size=3)# 8*9*13*13 
        self.conv3 = nn.Conv2d(in_channels=9,out_channels=9,kernel_size=3)# 8*9*13*13 
        self.fc1 = nn.Linear(9 * (3*grids[0]-4) * (3*grids[0]-4), 1)  
    def forward(self,x):
        x = fc.relu(self.conv1(x))
        x = fc.relu(self.conv2(x))
        x = fc.relu(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        return x

##加载模型
model = HeatMapCNN()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
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

print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)