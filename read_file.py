import numpy as np
import torch
import os
import torch.utils.data as Data
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

    # 将X和y转化为NumPy数组
    X = np.array(X)
    y = np.array(y)

    # 归一化图像数据（根据需要）
    X = X / 255.0  # 假设使用0-255的像素值

  