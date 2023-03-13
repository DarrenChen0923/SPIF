from aisf_jackie_library import *
import pandas as pd


outfilenum = 3
d = 5

# restructure_pointcloud_file("ybq/D4_{filenum}_completo.txt","ybq/f{filenum}_out.txt".format(filenum = outfilenum))
# restructure_pointcloud_file("ybq/D4_3_completo.txt","ybq/f3_out.txt")

finpath = "/home/durrr/phd/AISF_Jackie/MainFolder/fin_reg.txt"

# foutpath = "ybq/f2_out.txt"
foutpath = "/home/durrr/phd/AISF_Jackie/MainFolder/f{filenum}_out.txt".format(filenum = outfilenum)
# foutpath = "ybq/f1_out.txt"
# outputfilepath = "ybq/gridized_error_cloud2.txt"
outputfilepath = "/home/durrr/phd/AISF_Jackie/MainFolder/{size}mm_file/outfile{filenum}/gridized{size}mm_error_cloud{filenum}.txt".format(size = d, filenum = outfilenum)

# outputfilepath = "/Users/yangbingqian/Desktop/source/Colab-Notebooks/ybq/{size}mm_file/outfile{filenum}/gridized{size}mm_error_cloud1.txt".format(size = d, filenum = outfilenum) 
generate_gridized_cloud(finpath, foutpath, d, outputfilepath)

get_timeseries(outputfilepath,finpath,"/home/durrr/phd/AISF_Jackie/MainFolder//{size}mm_file/outfile{filenum}/trainingfile_{size}mm.txt".format(size = d, filenum = outfilenum), d)
# /home/durrr/phd/AISF_Jackie/MainFolder/10mm_file/outfile1/trainingfile_10mm.txt