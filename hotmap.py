import cv2 as cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

#Find the min and max value in all dataset
hot_map = []
def convert(num):

    new_num = int(num)

    return 171 + new_num 
# 找3个out文件的最大最小误差值
def readData(fnum,d):
    file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/1mm_file/outfile{fnum}/gridized1mm_error_cloud{fnum}.txt'
    with open(file_path,"r") as txtFile:
        line = txtFile.readline()
        while line:
            # print(line)
            line_split = line.split(",")

            if "nan" in line:
                line = txtFile.readline()
                continue

            if "center" in line:
                line = txtFile.readline()
                continue

            if "x" in line:
                line = txtFile.readline()
                continue
            
            if "y" in line:
                line = txtFile.readline()
                continue
            
            if "z" in line:
                line = txtFile.readline()
                continue

            error = float(line_split[3])
            
            hot_map.append(error)

            line = txtFile.readline()

# d_list = [5,10,15,20,30,40,50]
fnum_list = [1,2,3]


for fnum in fnum_list:
    # for d in d_list:
        readData(fnum,1)

min_value = np.min(hot_map)
max_value = np.max(hot_map)
print("Z:",(max_value-min_value))

#First: Draw F_in heatmap

#parameters
d = 20
fnum = 3
rotate = 270

in_data = np.zeros((342,342),np.float32)

#读取fin文件的数据
# file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/1mm_file/outfile{fnum}/gridized1mm_error_cloud{fnum}.txt'
file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/fin_reg.txt'
# file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/{d}mm_file/outfile{fnum}/gridized{d}mm_error_cloud{fnum}.txt'
with open(file_path,"r") as txtFile:
# with open("/Users/darren/资料/SPIF_DU/MainFolder/fin_reg.txt","r") as txtFile:
    line = txtFile.readline()
    while line:

        # print(line)
        
        line_split = line.split(",")

        if "nan" in line:
            line = txtFile.readline()
            continue

        if "center" in line:
            line = txtFile.readline()
            continue

        if "x" in line:
            line = txtFile.readline()
            continue
        
        if "y" in line:
            line = txtFile.readline()
            continue
        
        if "z" in line:
            line = txtFile.readline()
            continue

        x = convert(float(line_split[0]))
        y = convert(float(line_split[1]))
        error = float(line_split[2])
        
        in_data[x][y] = error

        line = txtFile.readline()


#create heatmap
# ax = sns.heatmap( out_data , linewidth = 0 , cmap = 'coolwarm',square = True ).invert_yaxis()
# plt.title( "Heat Map" )

#Min-Max标准化：
# def min_max_normalize(z_arr):
#     for i in range(len(z_arr)):
#         for j in range(len(z_arr[i])):
#             if z_arr[i][j] == 0.0:
#                 z_arr[i][j] = -1
#             else:
#                 z_arr[i][j] = (z_arr[i][j] - min_value) / (max_value - min_value)

#     return z_arr

in_data_temp = np.array(in_data)
# normalize_data = min_max_normalize(out_data)
# #保存fin热力图
plt.axis('off')
plt.set_cmap('Reds')
# plt.set_cmap('bwr')
plt.imshow(in_data_temp)
plt.savefig(f'/Users/darren/资料/SPIF_DU/Croppings/fin/heatmap.jpg',bbox_inches ='tight',pad_inches = 0)
plt.show()


#保存后的热力图大小为：369*369
#压缩热力图至342*342
crop_size = (342,342)
img = cv2.imread(f'/Users/darren/资料/SPIF_DU/Croppings/fin/heatmap.jpg')
img_new = cv2.resize(img,crop_size,interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/fin/heatmapresize.jpg",img_new)


#Second: Cropping 𝐹_𝑖𝑛 heatmap to 3*3 (e.g., 5mm -> 15*15) small figure
#Third: Calculate 𝐹_(1_𝑜𝑢𝑡) (2,3 as well) error
#Crop figure to 3*3 / 342*342
img = cv2.imread(f'/Users/darren/资料/SPIF_DU/Croppings/fin/heatmapresize.jpg')
image_copy = img.copy() 
imgheight=img.shape[0]
imgwidth=img.shape[1]
center = (imgwidth//2,imgheight//2)
M_1 = cv2.getRotationMatrix2D(center,-90,1.0)
image_copy = cv2.warpAffine(image_copy,M_1,(imgwidth,imgheight))

#读取所需要的out文件和grid size
out_data = np.zeros((342,342),np.float32)
#读取fout文件的数据
file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/{d}mm_file/outfile{fnum}/gridized{d}mm_error_cloud{fnum}.txt'
with open(file_path,"r") as txtFile:
    line = txtFile.readline()
    while line:

        # print(line)
        
        line_split = line.split(",")

        if "nan" in line:
            line = txtFile.readline()
            continue

        if "center" in line:
            line = txtFile.readline()
            continue

        if "x" in line:
            line = txtFile.readline()
            continue
        
        if "y" in line:
            line = txtFile.readline()
            continue
        
        if "z" in line:
            line = txtFile.readline()
            continue

        x = convert(float(line_split[0]))
        y = convert(float(line_split[1]))
        error = float(line_split[2])
        
        out_data[x][y] = error

        line = txtFile.readline()

#rotoate fout
rotate_data_90 =  list(zip(* out_data[::-1]))
rotate_data_180 =  list(zip(* rotate_data_90[::-1]))
rotate_data_270 =  list(zip(* rotate_data_180[::-1]))

#计算区间内平均误差
def calculate_error(x_start,x_end,y_start,y_end):
    total = 0
    count = 0
    error = 0 
    for row in range(x_start, x_end + 1):
        for col in range(y_start, y_end + 1):
            if rotate == 0:
                total += out_data[row][col]
            elif rotate == 90:
                total += rotate_data_90[row][col]
            elif rotate == 180:
                total += rotate_data_180[row][col]
            elif rotate == 270:
                total += rotate_data_270[row][col]
            count += 1          
            

    if count != 0:
        error =  total / count
    
    return error

# 3*3 so M = 3*d
M = 3 * d
N = 3 * d
x1 = 0
y1 = 0
 #将fin的热力图根据grid size切割成小图片
for y in range(0, imgheight, M):
    for x in range(0, imgwidth, N):
        if (imgheight - y) < M or (imgwidth - x) < N:
            break
             
        y1 = y + M
        x1 = x + N
 
        # check whether the patch width or height exceeds the image width or height
        if x1 >= imgwidth and y1 >= imgheight:
            x1 = imgwidth - 1
            y1 = imgheight - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
        elif y1 >= imgheight: # when patch height exceeds the image height
            y1 = imgheight - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
        elif x1 >= imgwidth: # when patch width exceeds the image width
            x1 = imgwidth - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
        else:
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
        #Save each patch into file directory
        if rotate == 0:
            cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm//saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            name = f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/images/"+str(x1)+"_"+str(y1)+".jpg"
            error_name = f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/labels/"+str(x1)+"_"+str(y1)+".txt"
        else:
            cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/rotate/{rotate}/saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            name = f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/rotate/{rotate}/images/"+str(x1)+"_"+str(y1)+".jpg"
            error_name = f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/rotate/{rotate}/labels/"+str(x1)+"_"+str(y1)+".txt"
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)  
        #根据所选择的out文件和不同的grid size计算误差并保存
        # 旋转后路径
        with open(error_name,"w") as file:
            error = calculate_error(x,x+N,y,y+M)
            file.write(str(error))
        cv2.imwrite(name,tiles)
       
print("Finish generating file")
cv2.imshow("Patched Image2",img)
if rotate == 0:
    cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/patched{fnum}.jpg",img)
else:
    cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/rotate/{rotate}/patched{fnum}.jpg",img)
cv2.waitKey()
cv2.destroyAllWindows()


#Fourth: Generate corresponding data <small figure, error>


import os
from sklearn.model_selection import train_test_split

# 设置图像文件夹和标签文件夹的路径
image_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/images'
label_folder = f'/Users/darren/资料/SPIF_DU/Croppings/f{fnum}_out/{d}mm/labels'

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
        img = img.resize((224, 224))  # 例如，将图像调整为 224x224 像素
        img = np.array(img)  # 将图像转化为NumPy数组

    # 将图像数据和标签添加到列表
    X.append(img)
    y.append(label)

# 将X和y转化为NumPy数组
X = np.array(X)
y = np.array(y)

# 归一化图像数据（根据需要）
X = X / 255.0  # 假设使用0-255的像素值

# 划分数据集为训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model. Go to heatmap_cnn.py