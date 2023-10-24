import sys
import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
# %matplotlib notebook
from math import hypot
from itertools import combinations

f_error = pd.read_csv('/Users/yangbingqian/Desktop/SAIF/visualization/10mm_file/outfile1/gridized10mm_error_cloud1.txt')
# f_error = pd.read_csv('error.txt')
gin = pd.read_csv('fin_reg.txt')
f_error['error'].describe()

f_error

d = 2

# f_error = f_error[f_error['error'].notna()]

#dataframe多条件排序
#第二列是在第一列排序基础上继续排序，也就是当第一列值相同的时候再去排对应的第二列
#x在y从大到小（y:169 to -169）排列的基础上，从小到大（x:-169 to 169）排列
# f_error = f_error.sort_values(by = ['center_y','center_x'], ascending = (False, True))
# len(f_error[f_error['center_y']==-39])
# len(f_error[f_error['center_y']==163])
# f_error['cordinate_row'] = np.nan
# f_error['cordinate_col'] = np.nan

# f_error = f_error[abs(f_error['error']) <= 6] ###!!!!!!!!!!!!!!!!!!!!

# # plt.figure(figsize=(80,8))
# plt.plot(f_error['error'], marker = ".")
# plt.show()

d=10



x_axis = np.arange(-171,171,d)
y_axis = np.arange(-171,171,d)
X,Y = np.meshgrid(x_axis,y_axis)

plt.plot(X,Y, marker='|', markersize=10, color='black', linestyle='none',alpha=0.2)
plt.plot(X,Y, marker='_', markersize=10, color='black', linestyle='none',alpha=0.2)
plt.plot(f_error['center_x'],f_error['center_y'], marker='.',markersize=0.4, color='red', linestyle='none')
plt.plot(gin['x'],gin['y'], marker='.',markersize=0.04, color='grey', linestyle='none',alpha=0.4)
plt.gca().set_aspect('equal') #set same ratio for x and y axis
plt.show()

f = pd.read_csv('f1out_python.txt')
f_error = f_error.dropna()
plt.plot(X,Y, marker='|', markersize=10, color='black', linestyle='none',alpha=0.2)
plt.plot(X,Y, marker='_', markersize=10, color='black', linestyle='none',alpha=0.2)
plt.plot(f_error['center_x'],f_error['center_y'], marker='.',markersize=0.3, color='red', linestyle='none')
plt.plot(f['x'],f['y'], marker='.',markersize=0.04, color='black', linestyle='none',alpha=0.1)
plt.gca().set_aspect('equal') #set same ratio for x and y axis
plt.show()

gin.head()

gin["y"].min()

# # 左下xy，右下xy，左上xy，右上点xy, 中点xy,
# pd.options.mode.chained_assignment = None
# grid_cordinate = [list(i) for i in zip(X.flat, Y.flat,
#                                        (X+d_xl).flat, Y.flat,
#                                        X.flat, (Y+d_xl).flat,
#                                        (X+d_xl).flat, (Y+d_xl).flat)]
# for grid in grid_cordinate:
#     error_grid = f_error[(f_error['center_x'] >= grid[0])
#                          & (f_error['center_x'] < grid[2])
#                          & (f_error['center_y'] >= grid[1])
#                          & (f_error['center_y']< grid[5])]
#     print(len(error_grid))
#     print()

#center point 一共171行
# f_error['center_y'].value_counts()
# len(f_error['center_x'].value_counts())
# f_error
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3,
#                        ):
#     print(f_error)


#      print(f_error.sort_values(by = ['center_y','center_x'], ascending = (False, True))['center_y'].value_counts())
#     print(f_error['center_y'].value_counts(sort=True))

# len(f_error[f_error['center_y']>=170])
# f_error[(f_error['center_y'] >= 166) & (f_error['center_y'] < 171)] #170: 152
# f_error[(f_error['center_y'] > 161) & (f_error['center_y'] < 166 )] #168: 171
# f_error[(f_error['center_y'] > -3) & (f_error['center_y'] < -1)] #168: 171

# f_error[(f_error['center_y'] > 167) & (f_error['center_y'] < 168)] # nothing
# # f_error[(f_error['center_y'] > 165) & (f_error['center_y'] < 167)] #166: 171

# print(len(f_error['center_y'].value_counts()))
y_cord = int(gin["y"].max())+1
# print(y_cord)
row = 0
arr = []
arr_col = []

while y_cord >= (-int(gin["y"].max())+1):
#     print(y_cord)
    len_y = len(f_error[(f_error['center_y'] >= y_cord-d) & (f_error['center_y'] < y_cord)])
#     print(len_y)
    arr += [row]*len_y


    for col in range(0,len_y):
        arr_col += [col]


    y_cord -= d
    row += 1

# print(len(arr))
arr = pd.DataFrame(arr[::-1], columns = ["row"])

# print(len(arr_col))

f_error['row'] = arr

f_error['col'] = arr_col

# f_error

# len(f_error['center_y'])
# ls_point = list(zip(f_error['center_x'],f_error['center_y'], f_error['center_z']))

# def dist(point, centerpt):
#     xc,yc = centerpt
#     x,y = point
#     return hypot(xc-x, yc-y)

# ls_point = list(zip(f_error['center_x'],f_error['center_y'], f_error['center_z']))

# print([dist(*combo) for combo in combinations(ls_point,2)])



# ls_point = list(zip(f_error['center_x'],f_error['center_y']))
# distances_matrix = np.array([np.linalg.norm((item*np.ones((len(ls_point),len(item))))-ls_point,axis=1) for item in ls_point])

f_error

# f_error = f_error[f_error['error'].notna()]
f_error

row = 0
z_matrix = []
error_matrix = []

while row <= f_error['row'].max():
    pt = f_error[f_error['row']==row]
    grid_cordinate = list(pt['center_z'])
    z_matrix.append(grid_cordinate)

    error_cordinate = list(pt['error'])
    error_matrix.append(error_cordinate)
    row += 1

from itertools import zip_longest
# pd.DataFrame(point_matrix)
error_matrix = np.array(list(zip_longest(*error_matrix, fillvalue= np.nan)))
# z_matrix = np.array(list(zip_longest(*z_matrix, fillvalue= z_matrix[len(z_matrix)-1])))
z_matrix = np.array(list(zip_longest(*z_matrix, fillvalue= np.nan)))

# data_tf = tf.convert_to_tensor(point_matrix, np.float32)

# data_tf

error_matrix.shape

z_matrix.shape

def rolling_window(a, shape):
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def window2(arr, shape=(3, 3)):
    r_extra = np.floor(shape[0] / 2).astype(int)
    c_extra = np.floor(shape[1] / 2).astype(int)
    out = np.empty((arr.shape[0] + 2 * r_extra, arr.shape[1] + 2 * c_extra))
    out[:] = np.nan
    out[r_extra:-r_extra, c_extra:-c_extra] = arr
    view = rolling_window(out, shape)
#     print(rolling_window(out, shape))
    return view

# window2(z_matrix,(3,3))
error_sqmatrix = window2(error_matrix,(3,3))
z_sqmatrix = window2(z_matrix,(3,3))

# z_sqmatrix.shape
# output:(170, 170, 3, 3)

# z_sqmatrix

# error_sqmatrix

# model_x = []
# import itertools
# for row in z_sqmatrix:
#     out = []
#     for col in row:
# #         print(list(itertools.chain.from_iterable(col)))
#         out.append(list(itertools.chain.from_iterable(col)))
#     model_x.append(out)


# # model_x
# model_x = np.array(model_x)
# model_x.shape

# model_y = []

# for row in error_sqmatrix:
#     out = []
#     for col in row:
# #         print(col)
# #         print(col[1,1])
#         out.append(col[1,1])
# #         out.append(list(itertools.chain.from_iterable(col)))
#     model_y.append(out)

# model_y = np.array(model_y)
# model_y.shape

#中间点坐标是[大行，大列，9格的1行，9格的1列]
# z_sqmatrix[0,27]
#          [nan,             nan,             nan            ],
#          [ 0.00000000e+00, -6.69697143e-02, -1.75795500e-01],
#          [-4.27424118e-01, -6.39255909e-01, -7.36666286e-01]

# error_sqmatrix[0,27]
# array([[        nan,         nan,         nan],
#        [        nan, -4.13421728, -4.09667037],
#        [-3.95238046, -3.85273341, -3.81147901]])

# z_sqmatrix[0,28,1,1]
# output: -0.0669697142857142

# error_sqmatrix[0,28,1,1]
#output: -4.134217275460144

model_x = []
import itertools
for row in z_sqmatrix:
    for col in row:
#         print(list(itertools.chain.from_iterable(col)))
        model_x.append(list(itertools.chain.from_iterable(col)))


# model_x
# model_x = np.array(model_x)
# model_x

model_y = []

for row in error_sqmatrix:
    for col in row:
#     for col_idx in range(0,3):
#         print(list(itertools.chain.from_iterable(col)))
        model_y.append(list(itertools.chain.from_iterable(col))[4])

# model_x
model_y = np.array(model_y)

# model_x = np.nan_to_num(model_x)
# model_x

# model_y = np.nan_to_num(model_y,nan = f_error['error'].mean())
# model_x

model_x[0]# input of prediction model

model_y[0] #output of prediction model

#write trainningfile
f = open("trainning_file.txt", "w")
# f.write("data_input | label" + "\n" )
for i in range(0, len(model_x)):
    if "nan" not in str(model_x[i]) and "nan" not in str(model_y[i]):
        f.write(str(model_x[i]).replace('[','').replace(']','') + " | " + str(model_y[i]) + "\n")
    else:
        pass

#read result from above
dataset_x = []
dataset_y = []
with open('trainning_file.txt', 'r') as f:
    lines = f.readlines()
    # random.shuffle(lines)
    # print(lines[10])
    for line in lines:
        line = line.strip("\n")
        dataset_x.append(line.split("|")[0].split(","))
        dataset_y.append(line.split("|")[1])

outliers = pd.DataFrame(dataset_y)
outliers= outliers.apply(pd.to_numeric, errors='ignore')
outliers[0].describe()

"""## reshape dataset"""

def create_dataset(X, y, seq_len=1):
    features = []
    targets = []

    for i in range(0, len(X)):
        data = [[i] for i in X[i]] # 序列数据
        label = [y[i]] # 标签数据

        # 保存到features和labels
        features.append(data)
        targets.append(label)

    # 返回
    return np.array(features), np.array(targets)

# ③ 数据集切分
# 功能函数：基于新的特征的数据集和标签集，切分：X_train, X_test

def split_dataset(x, y, train_ratio=0.8):

    x_len = len(x) # 特征数据集X的样本数量
    train_data_len = int(x_len * train_ratio) # 训练集的样本数量

    x_train = x[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集

    x_test = x[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集

    # 返回值
    return x_train, x_test, y_train, y_test

def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=128):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)

model_x,model_y = create_dataset(model_x, model_y, seq_len=1)

if "nan" in str(model_y[0].tolist()):
    print("yep")

if "nan" not in str(model_x[0].tolist()) or "nan" not in str(model_y[0].tolist()):
        print("yep")

len(model_x)



x_train, x_test, y_train, y_test = split_dataset(model_x, model_y, train_ratio=0.8)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

train_batch_dataset = create_batch_dataset(x_train, y_train)
test_batch_dataset = create_batch_dataset(x_test, y_test, train=False)

test_batch_dataset

# list(test_batch_dataset.as_numpy_iterator())[0]

"""## LSTM Model"""

model = Sequential([
    layers.LSTM(8),
    layers.Dense(1)
])

file_path = "best_checkpoint.hdf5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)

model.compile(optimizer='adam', loss="mse") #or mae

history = model.fit(train_batch_dataset,
          epochs=10,
          validation_data=test_batch_dataset,
          callbacks=[checkpoint_callback])

