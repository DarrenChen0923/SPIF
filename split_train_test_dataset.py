from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import shutil

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [5]
version = 2

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
    image_paths = []

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
                # print(f"Average RGB value of the image f_num={f_num},degree={degree},d={d}: R={avg_r/255}, G={avg_g/255}, B={avg_b/255},z = {label}")
            # 这里可以添加图像预处理步骤，例如将图像调整为固定大小、归一化等
            # img = img.convert("L")
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))
        # img = img.reshape(1,15,15)
        # 将图像数据和标签添加到列表
        # put image and label into list
        image_paths.append(image_path)
        X.append(img)
        y.append(label)
    return image_paths,X,y
  

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
    image_paths = []

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
                # print(f"Average RGB value of the image f_num={f_num},degree={degree},d={d}: R={avg_r/255}, G={avg_g/255}, B={avg_b/255},z = {label}")
            # 这里可以添加图像预处理步骤，例如将图像调整为固定大小、归一化等
            # img = img.convert("L")
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))
        # 将图像数据和标签添加到列表
        # put image and label into list
        image_paths.append(image_path)
        X.append(img)
        y.append(label)
    return image_paths,X,y
  

# f_num = 3
# d = 5
# X,y = read_data(3,5)
X = []  # 用于存储图像数据 store images
y = []  # 用于存储标签 store labels
image_paths = []

for fum in fums:
    for grid in grids:
        for degree in degrees:
            image_paths_grid_degree,X_fum_grid_degree,y_fum_grid_degree = read_data(fum,grid,degree)
            image_paths_grid_degree_flip,X_fum_grid_degree_flip,y_fum_grid_degree_flip = read_data_flip(fum,grid,degree)
            X= X + X_fum_grid_degree + X_fum_grid_degree_flip
            y= y + y_fum_grid_degree + y_fum_grid_degree_flip
            image_paths = image_paths + image_paths_grid_degree + image_paths_grid_degree_flip

Image_train, Image_test, y_train, y_test = train_test_split(image_paths,y,test_size=0.28,random_state=42)

def save_images_and_labels(images, labels, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image_path, label) in enumerate(zip(images, labels)):
        # destination_path = os.path.join(output_dir, f"images/{prefix}_image_{i+1}.png")
        # shutil.copy2(image_path,destination_path)
        label_path = os.path.join(output_dir, f"labels/{prefix}_{i+1}.txt")
        with open(label_path, 'w') as label_file:
            label_file.write(str(label))
        destination_path = os.path.join(output_dir, f"images/{prefix}_{i+1}.jpg")
        shutil.copy(image_path,destination_path)
    

# 保存训练集图像和标签
save_images_and_labels(Image_train, y_train, f'/Users/darren/资料/SPIF_DU/Croppings/version_2/train_dataset/{grids[0]}mm/', 'train')

# 保存测试集图像和标签
save_images_and_labels(Image_test, y_test, f'/Users/darren/资料/SPIF_DU/Croppings/version_2/test_dataset/{grids[0]}mm/', 'test')

