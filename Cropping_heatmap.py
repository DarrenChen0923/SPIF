import cv2 as cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

hot_map = np.zeros((342,342),np.float32)

def convert(num):
    new_num = int(num)
    return 171 + new_num 

#calculate error
def calculate_error(x,y):
    error = 0
    for i in range(x-3,x):
        for j in range(y-3,y):
            error+= hot_map[i][j]

    return error/9


d = 5
labels_path = "/Users/darren/资料/SPIF_DU/Croppings/{size}mm/labels/labels_{size}mm.txt".format(size = d)

f_labels = open(labels_path,"w+")
f_labels.write('end_x,'+'end_y,'+'error'+"\n")

with open("/Users/darren/资料/SPIF_DU/MainFolder/5mm_file/outfile3/gridized5mm_error_cloud3.txt","r") as txtFile:

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

img = cv2.imread('MainFolder/heatmap/outfile3/heatmap_5mm.png')
print(img.shape)


#Crop figure to 3*3 / 342*342
image_copy = img.copy() 
imgheight=img.shape[0]
imgwidth=img.shape[1]

M = 3 
N = 3
x1 = 0
y1 = 0
 
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
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif y1 >= imgheight: # when patch height exceeds the image height
            y1 = imgheight - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif x1 >= imgwidth: # when patch width exceeds the image width
            x1 = imgwidth - 1
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        else:
            #Crop into patches of size MxN
            tiles = image_copy[y:y+M, x:x+N]
            #Save each patch into file directory
            cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        name = "/Users/darren/资料/SPIF_DU/Croppings/5mm/images/"+str(x1)+"_"+str(y1)+".jpg"
        cv2.imwrite(name,tiles)
        error = calculate_error(x1,y1)
        f_labels.write(str(x1)+','+str(y1)+','+str(error)+"\n")
print("Finish generating file")
cv2.imshow("Patched Image",img)
cv2.imwrite("patched.jpg",img)
  
cv2.waitKey()
cv2.destroyAllWindows()




