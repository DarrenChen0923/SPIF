import cv2 as cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray


def flip_horizontal(matrix):
    return [row[::-1] for row in matrix]

#Find the min and max value in all dataset
hot_map = []
def convert(num):

    new_num = int(num)

    return 171 + new_num 
# 找3个out文件的最大最小误差值
# Find the maximum and minimum error values ​​of 3 out files
def readData(fnum):
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
        readData(fnum)

min_value = np.min(hot_map)
max_value = np.max(hot_map)
print("Z:",(max_value-min_value))

#First: Draw F_in heatmap

#parameters
version = 2

in_data = np.zeros((342,342),np.float32)
#读取fin文件的数据
# Read F_in 
file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/fin_reg.txt'
# file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/demostrador_caliente_v3.txt'
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
            
        in_data[x][y] = error

        line = txtFile.readline()

#create heatmap
in_data_temp = np.array(in_data)
# normalize_data = min_max_normalize(out_data)
# 保存fin热力图
# Svae F_in heatmap
plt.axis('off')
plt.set_cmap('Reds')
plt.imshow(in_data_temp)
plt.savefig(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/fin/heatmap.jpg',bbox_inches ='tight',pad_inches = 0)
# plt.show()

# 保存后的热力图大小为：369*369
# The size of the saved heatmap is: 369*369
# 压缩热力图至342*342
# Compress heat map to 342*342
crop_size = (342,342)
img = cv2.imread(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/fin/heatmap.jpg')
img_new = cv2.resize(img,crop_size,interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/fin/heatmapresize.jpg",img_new)

#Generate data method
def generate_data(d,fnum,rotate,version):
    #Second: Cropping 𝐹_𝑖𝑛 heatmap to 3*3 (e.g., 5mm -> 15*15) small figure
    #Third: Calculate 𝐹_(1_𝑜𝑢𝑡) (2,3 as well) error
    #Crop figure to 3*3 / 342*342
    img = cv2.imread(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/fin/heatmapresize.jpg')
    image_copy = img.copy() 
    image_flip = cv2.flip(img,1)
    imgheight=img.shape[0]
    imgwidth=img.shape[1]
    center = (imgwidth//2,imgheight//2)
    M_1 = cv2.getRotationMatrix2D(center,rotate,1.0)
    image_copy = cv2.warpAffine(image_copy,M_1,(imgwidth,imgheight))
    # 读取所需要的out文件和grid size
    # Store data
    out_data = np.zeros((342,342),np.float32)
    # 读取fout文件的数据
    # Read F_out 
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
            error = float(line_split[3])
            
            out_data[x][y] = error

            line = txtFile.readline()

    #rotoate fout
    rotate_data_90 =  list(zip(* out_data[::-1]))
    rotate_data_180 =  list(zip(* rotate_data_90[::-1]))
    rotate_data_270 =  list(zip(* rotate_data_180[::-1]))
    flipped_data = flip_horizontal(out_data)
    # 计算区间内平均误差
    # Calculate error
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

    #计算翻转的误差
    def calculate_error_flip(x_start,x_end,y_start,y_end):
        total = 0
        count = 0
        error = 0 
        for row in range(x_start, x_end + 1):
            for col in range(y_start, y_end + 1):
                total += flipped_data[row][col]
                count+=1
        if count != 0:
            error =  total / count
        return error
    # 3*3 so M = 3*d
    M = 3 * d
    N = 3 * d
    x1 = 0
    y1 = 0
    # 将fin的热力图根据grid size切割成小图片
    # Cropping heatmao according to grid size
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
                cropping = image_copy[y:y+M, x:x+N]
                flip = image_flip[y:y+M, x:x+N]
            elif y1 >= imgheight: # when patch height exceeds the image height
                y1 = imgheight - 1
                #Crop into patches of size MxN
                cropping = image_copy[y:y+M, x:x+N]
                flip = image_flip[y:y+M, x:x+N]
            elif x1 >= imgwidth: # when patch width exceeds the image width
                x1 = imgwidth - 1
                #Crop into patches of size MxN
                cropping = image_copy[y:y+M, x:x+N]
                flip = image_flip[y:y+M, x:x+N]
            else:
                #Crop into patches of size MxN
                cropping = image_copy[y:y+M, x:x+N]
                flip = image_flip[y:y+M, x:x+N]
            #Save each patch into file directory
            if rotate == 0:
                cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', cropping)
                rotate_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/images/"+str(x1)+"_"+str(y1)+".jpg"
                rorate_error_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/labels/"+str(x1)+"_"+str(y1)+".txt"
            else:
                cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/rotate/{rotate}/saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', cropping)
                rotate_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/rotate/{rotate}/images/"+str(x1)+"_"+str(y1)+".jpg"
                rorate_error_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/rotate/{rotate}/labels/"+str(x1)+"_"+str(y1)+".txt"
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)  
            # 根据所选择的out文件和不同的grid size计算误差并保存
            # Svae image
            # 旋转后路径
            # path after rorate 
            with open(rorate_error_name,"w") as file:
                error = calculate_error(x,x+N,y,y+M)
                file.write(str(error))
            cv2.imwrite(rotate_name,cropping)

            #Flip
            if rotate == 0:
                cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', flip)
                flip_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/images/"+str(x1)+"_"+str(y1)+".jpg"
                flip_error_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/labels/"+str(x1)+"_"+str(y1)+".txt"
            else:
                cv2.imwrite(f'/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/rotate/{rotate}/saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', flip)
                flip_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/rotate/{rotate}/images/"+str(x1)+"_"+str(y1)+".jpg"
                flip_error_name = f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/flip/rotate/{rotate}/labels/"+str(x1)+"_"+str(y1)+".txt"
                cv2.rectangle(image_flip, (x, y), (x1, y1), (0, 255, 0), 1)  
            with open(flip_error_name,"w") as file:
                error = calculate_error_flip(x,x+N,y,y+M)
                file.write(str(error))
            cv2.imwrite(flip_name,flip)
        
    print("Finish generating file")
    # cv2.imshow("Patched Image2",img)
    if rotate == 0:
        cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/patched{fnum}.jpg",img)
    else:
        cv2.imwrite(f"/Users/darren/资料/SPIF_DU/Croppings/version_{version}/f{fnum}_out/{d}mm/rotate/{rotate}/patched{fnum}.jpg",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


rotates = [0,90,180,270]
fums = [1,2,3]
grids = [5,10,15,20]

for fum in fums:
    for grid in grids:
        for rotate in rotates:
            generate_data(grid,fum,rotate,version)

#Fourth: Generate corresponding data <small figure, error>
#Implement in heatmap_cnn.py