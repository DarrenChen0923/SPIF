# SPIF (Single Point Increment Forming) Prediction Project

This is a project that uses artificial intelligence technology to predict rebound error for SPIF.
At present, the prediction of springback error has been realized through GRU and CNN.

The entire project is based on pytorch and scikit-learn libraries.

## Install Package

The code was tested under Ubentu 22.04 LTS, with python 3.6.13.

1. Clone project：

```bash
git clone git@github.com:DarrenChen0923/SPIF_DU.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```



## Download Dataset

We have split our data into `training data` and `testing data`, and provide the direct download link, you can downlaod them by the following link:

Training: [https://drive.google.com/file/d/1GrSz99zTtJSzKyFBNA6_NyS_af5rGplp](https://drive.google.com/file/d/1GrSz99zTtJSzKyFBNA6_NyS_af5rGplp)

Testing: [https://drive.google.com/file/d/1rBv1WHHg8GRF3JtkIBQo1r7EzCdW0fzy](https://drive.google.com/file/d/1rBv1WHHg8GRF3JtkIBQo1r7EzCdW0fzy)

Put them into `\croppings\version_2`, the final directory should look like the following:

```
\Croppings
    \version_2
        \train_dataset
        \test_dataset
```

These would be enough for reproducing the result. 

Alternatively, if you want to generate the heatmap by yourself, you need to following the instruction provided below:

1. Download the [original data files](https://drive.google.com/file/d/1XFJDcRiFojEpkhVEIZNHA1iPWChLrvum) and put them in the project root directory. An example directory should look like the following:

```
\SPIF_DU
    \Croppings
    \MainFolder
    \utils
    heatmap_cnn_new.py
    ...
```

2. Generating the heatmap:

```
python hotmap.py --project_root <project_root>
```

The <project_root> should be referenced as the path to the project folder `SPIF_DU`, e.g. if your project is at `/example/SPIF_DU`, pass `/example` as your argument.

3. Split the dataset for a certain grid size:

```
python split_train_test_dataset.py --project_root <project_root> --grid <5,10,15,20>
```

The grid size should be a numerical value referencing which grid size data you want to split. The availiable options are 5, 10 ,15, 20. Defult to be 15.



## Model
The CNN architecture consisted of (i) three 3*3 convolutional layers and (ii) a dense (fully connected) layer with 1 node for predicting springback error. The architecture is illustrated in Figure \ref{fig:cnnArchitecture}. A learning rate of $\alpha = 0.001$, a batch size $= 64$, and an epoch size $e = 1000$ were used. 

![My Image](Croppings/Architecutre.png)

## Training with Heatmap

To train the model for 

```bash
python heatmap_cnn_new.py --project_root <project_root> --grid <5,10,15,20>
```

The model will be saved into `trained_models` folder.


## Evaluation

To evaluate the model trained for a certain grid size, run the following:

```
python evalution_heatmap.py \\
--project_root <project_root> \\
--grid <5,10,15,20> \\
--load_model <model_name>
```

The <model_name> should be specified as the name of the one of the model in `trained_models` folder.

## Checkopoints

We provide the checkopoints for the trained models, feel free to test them!

| Grid Size | Model |
| :---         |     :---:      | 
| heatmap_cnn_5mm   | [download](https://drive.google.com/file/d/19OyUcionqjBT-ONE8axVqb8f7QMNU6Y-)    | 
| heatmap_cnn_10mm  | [download](https://drive.google.com/file/d/1zJPVon80R-4jOiqZORPSxatDcypdaCQz)    |
| heatmap_cnn_15mm  | [download](https://drive.google.com/file/d/1pouGvgYJ_8VkPIUubOuMizqsEeRe5iWu)   | 
| heatmap_cnn_20mm  | [download](https://drive.google.com/file/d/1bIp-YXgsotvh2CZtnzaLdM2CImsfw8K6)    |


## Result
The training result from the train dataset
| Grid Size | MAE | MSE | RMSE | R2 |
| :---         |     :---:      |          ---: |     :---:      |          ---: |
| 5mm   | 0.2337    | 0.1251   | 0.3531    | 0.9130      |
| 10mm  | 0.2079    | 0.0898   | 0.2913    | 0.9188      |
| 15mm  | 0.1270    | 0.0327   | 0.1804    | 0.9522      |
| 20mm  | 0.0924    | 0.0167   | 0.1287    | 0.9586      |

The test result from the test dataset
| Grid Size | MAE | MSE | RMSE | R2 |
| :---         |     :---:      |          ---: |     :---:      |          ---: |
| 5mm   | 0.2987    | 0.2029   | 0.4495    | 0.8522      |
| 10mm  | 0.3477    | 0.2562   | 0.4963    | 0.7663      |
| 15mm  | 0.1895    | 0.0869   | 0.2620    | 0.8913      |
| 20mm  | 0.1408    | 0.0384   | 0.1952    | 0.9298      |

