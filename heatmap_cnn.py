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

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [20]
version = 2

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
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/labels'
    else:
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/rotate/{degree}/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/rotate/{degree}/labels'
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
    count = 0
    for image_path in image_files:
        count+=1
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
            if count == 5:
                image = img.convert("RGB")

                # 获取图像的像素数据
                pixels = list(image.getdata())

                total_r, total_g, total_b = 0, 0, 0
                total_pixels = len(pixels)

                for r, g, b in pixels:
                    total_r += r
                    total_g += g
                    total_b += b

                avg_r = total_r // total_pixels
                avg_g = total_g // total_pixels
                avg_b = total_b // total_pixels


                # print(f"Average RGB value of the image f_num={f_num},degree={degree},d={d}: R={avg_r/255}, G={avg_g/255}, B={avg_b/255},z = {label}")
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
  

def read_data_flip(f_num,d,degree):
    if degree == 0:
        # 设置图像文件夹和标签文件夹的路径
        # set Image file and label file path
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/labels'
    else:
        image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/rotate/{degree}/images'
        label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/rotate/{degree}/labels'
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
    count = 0
    for image_path in image_files:
        count+=1
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
            if count == 5:
                image = img.convert("RGB")

                # 获取图像的像素数据
                pixels = list(image.getdata())

                total_r, total_g, total_b = 0, 0, 0
                total_pixels = len(pixels)

                for r, g, b in pixels:
                    total_r += r
                    total_g += g
                    total_b += b

                avg_r = total_r // total_pixels
                avg_g = total_g // total_pixels
                avg_b = total_b // total_pixels


                # print(f"Average RGB value of the image f_num={f_num},degree={degree},d={d}: R={avg_r/255}, G={avg_g/255}, B={avg_b/255},z = {label}")
            # 这里可以添加图像预处理步骤，例如将图像调整为固定大小、归一化等
            # img = img.convert("L")
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))
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
            X_fum_grid_degree_flip,y_fum_grid_degree_flip = read_data_flip(fum,grid,degree)
            X= X + X_fum_grid_degree + X_fum_grid_degree_flip
            y= y + y_fum_grid_degree + y_fum_grid_degree_flip

# 将X和y转化为NumPy数组
# transfer X and y into numpy array
X = np.array(X)
y = np.array(y)


# 归一化图像数据
# Normalise
X = X / 255.0  # 使用0-255的像素值
def normalize_mean_std(image):
    mean = np.mean(image)
    stddev = np.std(image)
    normalized_image = (image - mean) / stddev
    return normalized_image

X = normalize_mean_std(X)
batch = 64

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

# #Cominbe dataset
# train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
# val_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)

# #Create dataset loader

# train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
# val_data_loader = Data.DataLoader(val_dataset, batch_size=batch, shuffle=False)

result_file_path = f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/result/{grids[0]}mm/tencrosvalidationresult_validate.txt'


kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    # # 获取训练集和测试集
    # X_train, X_test = X_train_tensor[train_idx], X_train_tensor[test_idx]
    # y_train, y_test = y_train_tensor[train_idx], y_train_tensor[test_idx]

    # 将剩下的9份按照8:2的比例划分为训练集和验证集
    # Divide the remaining 9 parts into training sets and validation sets in a ratio of 8:2
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
    # Create model
    model = HeatMapCNN()
    # print(model)
    # 定义损失函数和优化器
    # Defein optimizer and criterion
    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) #lr
    # 记录训练和测试过程中的损失
    # store loss during train and test
    train_losses = []  # 训练损失 train loss
    test_losses = []  # 测试损失 test loss

    # 训练模型
    # train model
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
            # Checkt Gradient
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
    # After training is completed,  use the model for prediction
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
    # Store test data result
    with open(result_file_path,"a") as file:
        if fold == 0:
            file.write("Model: "+str(model))
            file.write("\nLearning rate: " + str(learning_rate))
            file.write("\nEpoch: " + str(num_epochs))
        file.write("\nFold: "+str(fold+1))
        file.write("\nMAE: "+str(mae))
        file.write("\nMSE: "+str(mse))
        file.write("\nRMSE: "+str(rmse))
        file.write("\nR2: "+str(r2)+"\n")
    # plt.figure()
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # # plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # 打印当前折数和各集合的大小
    print(f"Fold {fold + 1} - Train: {len(X_train)}, Validation: {len(X_val_fold)}, Test: {len(X_test)}")


#保存模型
# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
},  f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/models/model_epo{num_epochs}_batch{batch}_lr{learning_rate}_grid{grids[0]}_version{version}.pth')

# validation的结果
# Result of validation
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

with open(result_file_path,"a") as file:
    file.write("Valuation Result: ")
    file.write("\nMAE: "+str(mae))
    file.write("\nMSE: "+str(mse))
    file.write("\nRMSE: "+str(rmse))
    file.write("\nR2: "+str(r2)+"\n")

# # 定义模型和优化器
# model = MyModel()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 加载模型的状态字典和优化器的状态字典
# checkpoint = torch.load('trained_model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # 将模型设置为评估模式（不启用 Dropout 等）
# model.eval()