# SPIF(Single Point Increment Forming) Prediction Project

This is a project that uses artificial intelligence technology to predict rebound error for SPIF.
At present, the prediction of springback error has been realized through GRU and CNN.

The entire project is based on pytorch and scikit-learn libraries.

## Install

1. Clone project：

```bash
git clone git@github.com:DarrenChen0923/SPIF_DU.git
```

2. Install scikit-learn:
```bash
pip install scikit-learn
```

3. Install Pytorch:
```bash
pip install torch torchvision torchaudio
```
There are also some libraries commonly used in artificial intelligence that need to be installed (numpy, matplotlib, etc.). You can install them according to the prompts.

## Dataset
The entire project has 4 sets of original data, namely D4_1_completo.txt, D4_2_completo.txt, D4_3_completo.txt and fin_reg.txt in MainFolder. Fin_reg is called Fin and the remaining three are called Fout.

Process raw data according to different project requirements

1. Heatmap

Cropping image data in a folder called Croppings.
In this folder have four sub-folder: f1_out,f2_out,f3_out and fin. Coresponding to four raw data.

The path to the data used to generate the Fout heat map
```bash
.../MainFolder/{d}mm_file/outfile{fnum}/gridized{d}mm_error_cloud{fnum}.txt

d: Grid size (5,10,15,20)
fum: Fout number (1,2,3)
```

The path to the data used to generate the Fin heat map
```bash
.../MainFolder/fin_reg.txt
```


## Run Code

1. Generate Heatmap

```bash
Generate Fin file heatmap
file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/fin_reg.txt'

Generate Fout file heatmap
# file_path = f'/Users/darren/资料/SPIF_DU/MainFolder/{d}mm_file/outfile{fnum}/gridized{d}mm_error_cloud{fnum}.txt'
d: Grid size (5,10,15,20)
fum: Fout number (1,2,3)

python3 hotmap.py
```
Change to right directory according to your machine.

2. Run CNN model

```bash
python3 heatmap_cnn.py

```

3. Draw figure(line chart)

```bash
python3 draw_fig.py
```

