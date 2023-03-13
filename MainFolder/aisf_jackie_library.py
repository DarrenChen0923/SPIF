import numpy as np
from numpy.core.fromnumeric import resize
import pandas as pd
import math
import tensorflow as tf
import itertools
from scipy.linalg import solve
import statsmodels.api as sm 
from sklearn import datasets, linear_model
from tensorflow import keras
from tensorflow.python.keras import Sequential, layers, utils
from math import hypot
from itertools import combinations
from sklearn.model_selection import KFold

def delete_tab(oldfilename, newfilename):
    """delete spaces and replace with newline"""
    f_v3 = open(newfilename, "w+")

    with open(oldfilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip(" ")
            f_v3.write(' '.join(line.split()) + '\n')


# def numerical_txtfile(file):
#     """
#     Make sure a txt file can be read into a numerical file 
#     ignore any error. ( potential unsafe)
#     """
#     file = file.apply(pd.to_numeric, errors='ignore')
#     return file


#generate center point 
def generate_gridcordinate(d, x_max, x_min, y_max, y_min):
    """
    Generate a grid within size d*d for each point in XY mesh
    This fails on the edge case as the size is to upper right. 
    So i reduced to the top right corner.
    """ 
    # mm in this case
    #use gin to generate grid
    x_axis = np.arange(x_min,x_max-d,1)#sheng cheng deng cha shu lie 
    y_axis = np.arange(y_min,y_max-d,1)
    X,Y = np.meshgrid(x_axis,y_axis)

    # 左下xy，右下xy，左上xy，右上点xy, 中点xy
    pd.options.mode.chained_assignment = None
    grid_cordinate = [list(i) for i in zip(X.flat, Y.flat,
                                    (X+d).flat, Y.flat,
                                    X.flat, (Y+d).flat,
                                    (X+d).flat, (Y+d).flat,
                                    (X+d/2).flat, (Y+d/2).flat)]
    return grid_cordinate

#TODO ask why  0,2,1,5?
def find_points_in_grid(pointcloud,grid):
    """
    Find points in the grid
    The inputs seems to be the whole list
    The constraint seems to grid 0 2 1 5? why?
    """
    grid_pointcloud = pointcloud[
        (pointcloud['x'] >= grid[0]) 
        & (pointcloud['x'] < grid[2]) 
        & (pointcloud['y'] >= grid[1]) 
        & (pointcloud['y']< grid[5])]
    return grid_pointcloud


def restructure_pointcloud_file(oldfilename, newfilename):
    f1 = open(newfilename, "w+")
    f1.write("x,""y,"+"z"+"\n")

    with open(oldfilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            f1.write(line.replace(" ",",").strip()+"\n")

def getplane_ols(x_list, y_list, z_list):
    if len(x_list) < 3: 
        return [], []
    else: 
        X1=np.column_stack((x_list,y_list))
        X1=sm.add_constant(X1)
        model = sm.OLS(z_list, X1)
        result = model.fit()
        norm_vector = [result.params[1],result.params[2],-1]
        intersect = result.params[0]

        return norm_vector, intersect

def getplane_ols_new( xs, ys, zs):
    """A direct implementation of OLS to check the above"""
    l = len(xs)
    if l != len(ys) or l != len(zs):
        return []
    else:
        A = np.matrix(np.column_stack((np.ones(l),xs, ys)))
        b = np.matrix(zs).T
        fit = (A.T * A).I * A.T * b
        return fit.A1

def solve_for_intersection(norm_in, norm_out, cin, cout, intersect_out):
    """Solve for intersection between in and out planes/
    What is cin? cout is not used? 
     yes, cout is not used, and cin is the centerpoint of CAD file
    what is intersect_out?
     it is the intersect between normal vector from gin and gout plane
    do we need the planes to overlap or intercept or parallel?
     intersect between normal vector from gin and gout plane
    
    TODO: the logic to decide negative and positive error has to be discussed"""
    if len(norm_in) == 3 and len(norm_out) == 3:
        lhs = np.array([[norm_in[1]/norm_in[0],-1,0],
                        [0,-1,norm_in[1]/norm_in[2]],
                        [norm_out[0],norm_out[1],-1]])
        rhs = np.array([[((norm_in[1]*cin[0])/norm_in[0]) - cin[1]],
                        [((cin[2]*norm_in[1])/norm_in[2]) - cin[1]],
                        [-intersect_out]])
        try:
            result = solve(lhs,rhs)
            if result[2] - cin[2] >= 0:
                error = math.sqrt((result[0] - cin[0])**2 + (result[1] - cin[1])**2 + (result[2] - cin[2])**2)
            else:
                error = -math.sqrt((result[0] - cin[0])**2 + (result[1] - cin[1])**2 + (result[2] - cin[2])**2)
        
            return error,result
        
        except:
            return np.nan, np.nan
            
    else:
        return np.nan, np.nan



def generate_gridized_cloud(finpath,foutpath, gridsize, outputfilepath):
    #TODO: the gridsize has to be changed to see how LSTM work for each size
    
# Parameters 
  cad_fileName = finpath  #change to your own path
  realsheet_fileName = foutpath #change to your own path
  """ set *d (mm)* the length of side of grid and create grid"""
  gridsize  = gridsize

  #read files needed
  gin = pd.read_csv(cad_fileName, header = 0, dtype = {"x":float, "y": float, "z": float})
  gout = pd.read_csv(realsheet_fileName, header = 0, dtype = {"x":float, "y": float, "z": float})

#   gout = numerical_txtfile(gout)
#   gin = numerical_txtfile(gin)


  """set d (mm) the length of side of grid and create grid"""
  grid_cordinate = generate_gridcordinate(gridsize, gin['x'].max(), gin["x"].min(), gin["y"].max(), gin["y"].min())

  f_error = open(outputfilepath, "w+")
  f_error.write('center_x,'+'center_y,'+'center_z,'+'error'+"\n")
  idx = 0
  print("start generating file....")
  for grid in grid_cordinate:
      
      gout_grid = find_points_in_grid(gout, grid)
      gin_grid = find_points_in_grid(gin, grid)
      
      #embed grid index to each point
      gout_grid["grid_idx"] = idx 
      gin_grid["grid_idx"] = idx

      #center point cordinates xyz
      gout_center_xyz = [grid[8],grid[9],gout_grid["z"].mean()]
      gin_center_xyz = [grid[8],grid[9],gin_grid["z"].mean()]
      
      #拟合平面得到normal vector
      plane_gout_norm, b0_out = getplane_ols(list(gout_grid['x']),list(gout_grid['y']),list(gout_grid['z']))
      plane_gin_norm, b0_in = getplane_ols(list(gin_grid['x']),list(gin_grid['y']),list(gin_grid['z']))
      
      #solve the intersection point and calculate error
      solu, intersect_pt = solve_for_intersection(plane_gin_norm,plane_gout_norm,gin_center_xyz,gout_center_xyz, b0_out)
    #   print("center gin:", gin_center_xyz)
    #   print("gout point:",gout_grid)
    #   print("gin point:",gin_grid)
    #   print("real part intersect:", b0_out)
    #   print("cad_plane intersect: ", b0_in)
    #   print("plane_gout_norm vector:",plane_gout_norm)
    #   print("plane_gin_norm vector:", plane_gin_norm)
    #   print("intersect_pt:" , intersect_pt)
    #   print("error result:", solu)
    #   print("end")
    #   print("-----------")
    #   print()
    #   print()
      '''unmute below line to check outliers if nessecery'''
    #   getOutliers(solu,gout_grid, gin_grid,plane_gout_norm,plane_gin_norm,intersect_pt)

      f_error.write(str(gin_center_xyz[0]) +','+ str(gin_center_xyz[1])+',' + str(gin_center_xyz[2]) +','+ str(solu) + '\n')
      
      idx += 1

  print("finish generating gridized cloud file")


def getOutliers(solu,gout_grid, gin_grid,plane_gout_norm,plane_gin_norm,intersect_pt):
    '''check outliers from above and print out as output'''

    if abs(solu) >= 10: #cath outliers
        print("gout point:",gout_grid)
        print("gin point:",gin_grid)
        print("plane_gout_norm vector:",plane_gout_norm)
        print("plane_gin_norm vector:", plane_gin_norm)
        print("intersect_pt:" , intersect_pt)
        print("error result:", solu)
        print("end")
        print("-----------")
        print()
        print()
    else:
        pass


##prepare data function
def create_dataset(X, y):
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



def split_dataset(x, y, train_ratio=0.8):
    '''split trainning and testing dataset
    TODO: there could be a better and more randomn way to do split ?  '''
    x_len = len(x) # 特征数据集X的样本数量
    train_data_len = int(x_len * train_ratio) # 训练集的样本数量
    
    x_train = x[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集
    
    x_test = x[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集
    
    # 返回值
    return x_train, x_test, y_train, y_test



def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=128):
    '''not used, but is creating batch for LSTM model to execute faster'''
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 数据封装，tensor类型
    if train: # training set
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # testing set
        return batch_data.batch(batch_size)



def preparedata_LSTM(input_lst, lable_lst, splitratio):
    '''prepare data to fit lstm model, return training and testing dataset'''
    lable = [float(y) for y in lable_lst]
    input_x = []
    for grp in input_lst:
        input_x.append([float(z) for z in grp])
    
    input_x,lable = create_dataset(input_x, lable)
    x_train, x_test, y_train, y_test = split_dataset(input_x, lable, train_ratio = splitratio)
    return x_train, x_test, y_train, y_test



#create window
def rolling_window(a, shape):
    '''generate 3*3 window and let it sliding to get each input time series'''
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

def get_timeseries(gridcouldfilepath, ginfilepath,newpath, d):
    '''return LSTM model input file model_x, and label file model_y'''
    #Load file
    f_error = pd.read_csv(gridcouldfilepath)
    gin = pd.read_csv(ginfilepath,header = 0,dtype = {"x":float, "y": float, "z": float})
    y_cord = int(gin["y"].max())+1

    row = 0
    arr = []
    arr_col = []

    #TODO: check loop logic 

    while y_cord >= (-int(gin["y"].max())+1):
    #     print(y_cord)
        len_y = len(f_error[(f_error['center_y'] >= y_cord-d) & (f_error['center_y'] < y_cord)])
        # print(len_y)
        arr += [row]*len_y

        
        for col in range(0,len_y):
            arr_col += [col]
            
        
        y_cord -= d
        row += 1

    arr = pd.DataFrame(arr[::-1], columns = ["row"])

    f_error['row'] = arr
    f_error['col'] = arr_col


    #according to row_col label to generate zvalue_matrix
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
    

    error_sqmatrix = window2(error_matrix,(3,3))
    z_sqmatrix = window2(z_matrix,(3,3))

    #input
    model_x = []
    for row in z_sqmatrix:
        for col in row:
            model_x.append(list(itertools.chain.from_iterable(col)))
    
    #label
    model_y = []

    for row in error_sqmatrix:
        for col in row:
            model_y.append(list(itertools.chain.from_iterable(col))[4])
    model_y = np.array(model_y)
    
    f = open(newpath, "w+")
    # f.write("data_input | label" + "\n" )
    for i in range(0, len(model_x)):
        # print(newpath)
        if "nan" not in str(model_x[i]) and "nan" not in str(model_y[i]):
            f.write(str(model_x[i]).replace('[','').replace(']','') + " | " + str(model_y[i]) + "\n")
        else:
            pass

#load model and do comparison between diff grid size reuslt
def load_model_from_path(gsize,trainfilenum,shuffle,batch):
  print('LSTM Model build base on file' + str(trainfilenum))
  modelpath = '/content/drive/MyDrive/ColabNotebooks/{size}mm_file/outfile{fnum}/Model_shuff{shuff}_bch{batchsize}.h5'.format(
      size = gsize, fnum = trainfilenum, shuff = str(shuffle), batchsize = batch)
  model = load_model(modelpath)
  return model

def get_prediction_result(model, test_x, test_lable,testfilenum):
  print("predicting base on test outfile{testf}....".format(testf = testfilenum))
  y_pred = model.predict(test_x, verbose=1)
  score = r2_score(test_lable, y_pred)
  print("RMSE:", mean_squared_error(test_lable, y_pred, squared=False))
  print("MAE:",mean_absolute_error(test_lable, y_pred))
  print("MSE:",mean_squared_error(test_lable, y_pred))
  print("Rsquare:", score)
  print("___finished___")


def get_model_history(gsize,trainfilenum,shuffle,batch):
  historypth = '/content/drive/MyDrive/ColabNotebooks/{size}mm_file/outfile{fnum}/Model_shuff{shuff}_bch{batchsize}.pickle'.format(
      size = gsize, fnum = trainfilenum, shuff = str(shuffle), batchsize = batch)
  with open(historypth, 'rb') as file_pi:
    history=pickle.load(file_pi)
  return history



def show_loss_pic(history):
  print('LSTM Model Loss history when build base on file' + str(trainfilenum))
  plt.figure(figsize=(16,8))
  plt.plot(history['loss'], label='train loss')
  plt.plot(history['val_loss'], label='val loss')
  plt.title("LOSS")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()


def get_testfile(gsize,testfilenum,shuffle,batch):
  testfilepath = '/content/drive/MyDrive/ColabNotebooks/{size}mm_file/outfile{filenum}/trainingfile_{size}mm.txt'.format(size = gsize, filenum = testfilenum)
  dataset_x = []
  dataset_y = []
  with open(testfilepath, 'r') as f_test:
      lines = f_test.readlines()
      for line in lines:
          line = line.strip("\n")
          dataset_x.append(line.split("|")[0].split(","))
          dataset_y.append(line.split("|")[1])
  lable = [float(y) for y in dataset_y]
  input_x = []
  for grp in dataset_x:
      input_x.append([float(z) for z in grp])
      
  test_x,test_lable = create_dataset(input_x, lable)
  
  return test_x, test_lable


def start_main(trainfilenum,testfile1num,testfile2num,batch,gsize,shuffle):
  testa_x, testa_y = get_testfile(gsize,testfile1num,shuffle,batch)
  testb_x, testb_y = get_testfile(gsize,testfile2num,shuffle,batch)

  model = load_model_from_path(gsize,trainfilenum,shuffle,batch)

  history = get_model_history(gsize,trainfilenum,shuffle,batch)
  show_loss_pic(history)

  get_prediction_result(model,testa_x, testa_y,testfile1num)
  print()
  get_prediction_result(model,testb_x, testb_y,testfile2num)
