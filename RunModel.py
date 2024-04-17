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
from PIL import Image
import heatmap_cnn as cnn


degrees = [0,90,180,270]
fums = [1]
grids = [5]


# 自定义排序函数
def numeric_sort(file_name):
    # 提取文件名中的数字部分作为排序关键字
    file_name_parts = file_name.split('/')[-1].split('_')
    file_name_parts[-1] = file_name_parts[-1].split('.')[0]  # 去掉文件后缀
    return int(file_name_parts[0]), int(file_name_parts[1])

def read_data(f_num,d,degree):
    if degree == 0:
        # 设置图像文件夹和标签文件夹的路径
        # set Image file and label file path
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/labels'
    else:
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/rotate/{degree}/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{f_num}_out/{d}mm/rotate/{degree}/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

   # 按照数字排序文件列表
    image_files = sorted(image_files, key=numeric_sort)
    # image_files.sort(key=lambda x:int(x.split('.'[0])))

    # 创建空的训练数据列表，用于存储图像和标签
    # Create empty list to store iamges and lables
    X = []  # 用于存储图像数据 store images
    y = []  # 用于存储标签 store lables

    # 遍历图像文件列表
    # Iterate Images
    # count = 0
    for image_path in image_files:
        # count+=1
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
            # if count == 72:
            #     image = img.convert("RGB")

            #     # 获取图像的像素数据
            #     pixels = list(image.getdata())

            #     total_r, total_g, total_b = 0, 0, 0
            #     total_pixels = len(pixels)

            #     for r, g, b in pixels:
            #         total_r += r
            #         total_g += g
            #         total_b += b

            #     avg_r = total_r // total_pixels
            #     avg_g = total_g // total_pixels
            #     avg_b = total_b // total_pixels


            #     print(f"Average RGB value of the image f_num={f_num},degree={degree},d={d}: R={avg_r/255}, G={avg_g/255}, B={avg_b/255},z = {label}")
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
  

# f_num = 3
# d = 5
# X,y = read_data(3,5)
X = []  # 用于存储图像数据 store images
y = []  # 用于存储标签 store labels

for fum in fums:
    for grid in grids:
        for degree in degrees:
            X_fum_grid_degree,y_fum_grid_degree = read_data(fum,grid,degree)
            X+=X_fum_grid_degree
            y+=y_fum_grid_degree

# 将X和y转化为NumPy数组
# transfer X and y into numpy array
X = np.array(X)
y = np.array(y)


# 归一化图像数据
# Normalise
X = X / 255.0  # 使用0-255的像素值
batch = 8

# 划分数据集为训练集和验证集
# Divide dataset into train set and validation set
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


# 加载模型参数
model = cnn.HeatMapCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('model_epo1000_batch8_lr2e-06.pth'))
checkpoint = torch.load('trained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


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