# import torch
# from torchvision import transforms
# from PIL import Image

# # 1. 定义转换
# transform = transforms.Compose([
#     transforms.ToTensor(),            # 转换为张量
# ])

# # 2. 加载单张图片
# image_path = "/Users/darren/资料/SPIF_DU/Croppings/f1_out/5mm/images/45_120.jpg"
# image = Image.open(image_path)

# # 3. 进行预处理
# image_tensor = transform(image)

# # 4. 显示结果
# print(image_tensor.shape)  # 查看张量形状

import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt

input_file_path = "/Users/darren/资料/SPIF_DU/MainFolder/demostrador_caliente_v3.txt"
output_file_path = "/Users/darren/资料/SPIF_DU/MainFolder/demostrador_caliente_v3_new.txt"

# 打开输入文件和输出文件
with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    # 读取输入文件的内容
    file_content = input_file.read()
    # 将逗号替换为空格
    modified_content = file_content.replace(" ", ",")
    # 将修改后的内容写入输出文件
    output_file.write(modified_content)

print("文件处理完成！")