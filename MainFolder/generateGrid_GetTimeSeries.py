from aisf_jackie_library import *
import pandas as pd


outfilenum = 1
d = 1


# finpath = "/home/durrr/phd/SPIF_DU/MainFolder/fin_reg.txt"
finpath =  "/Users/darren/资料/SPIF_DU/MainFolder/fin_reg.txt"

foutpath =  "/Users/darren/资料/SPIF_DU/MainFolder/f{filenum}_out.txt".format(filenum = outfilenum)
# foutpath = "/home/durrr/phd/SPIF_DU/MainFolder/f{filenum}_out.txt".format(filenum = outfilenum)
# outputfilepath = "/home/durrr/phd/SPIF_DU/MainFolder/{size}mm_file/outfile{filenum}/gridized{size}mm_error_cloud{filenum}_overlapping_3.txt".format(size = d, filenum = outfilenum)
outputfilepath = "/Users/darren/资料/SPIF_DU/MainFolder/{size}mm_file/outfile{filenum}/gridized{size}mm_error_cloud{filenum}.txt".format(size = d, filenum = outfilenum)

generate_gridized_cloud(finpath, foutpath, d, outputfilepath)

get_timeseries(outputfilepath,finpath,"/home/durrr/phd/SPIF_DU/MainFolder//{size}mm_file/outfile{filenum}/trainingfile_{size}mm_overlapping_3.txt".format(size = d, filenum = outfilenum), d)