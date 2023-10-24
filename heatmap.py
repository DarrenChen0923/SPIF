import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray

hot_map = np.zeros((342,342),np.float32)

def convert(num):
    new_num = int(num)
    return 171 + new_num 

with open("/Users/darren/资料/SPIF_DU/MainFolder/50mm_file/outfile3/gridized50mm_error_cloud3.txt","r") as txtFile:

    line = txtFile.readline()
    while line:
        line_split = line.split(",")
        if "nan" in line:
            line = txtFile.readline()
            continue
        if "center" in line:
            line = txtFile.readline()
            continue

        x = convert(float(line_split[0]))
        y = convert(float(line_split[1]))
        error = float(line_split[3])        
        hot_map[x][y] = error

        line = txtFile.readline()

#create heatmap
plt.axis('off')
# map_vir = cm.get_cmap('Reds')
# plt.imshow(hot_map,cmap=map_vir)
# plt.show()
# plt.savefig('heatmap_40mm.png',bbox_inches = 'tight',pad_inches = 0,transparent = False)



#网格形式切割数据. 讲数据以折线图表示，然后转换为numpy数组数据. 342 能同时被2、9整除。So 选择3*3的滑动窗口
#计算误差
def calculate_error(data):
    error = 0
    for i in range(len(data)):
        error+=data[i]

    return error/len(data)



