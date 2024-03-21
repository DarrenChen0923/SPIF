import matplotlib.pyplot as plt

# 假设你有这样的数据 20mm
# mae_values = [0.007978464, 0.009161059, 0.00881325, 0.007934828, 0.008498118, 0.0076440196, 0.007022226, 0.008286227, 0.0076667094, 0.008117117]
# mse_values = [0.000133446, 0.00013073, 0.00012039307, 0.00012185566, 0.00013661105, 0.00010233594, 0.000103856706, 0.00012835687, 0.000106185624, 0.00012818261]
# rmse_values = [0.011551883, 0.0114337215, 0.010972378, 0.011038825, 0.011688073, 0.010116123, 0.010191011, 0.011329469, 0.010304641, 0.011321776]
# r2_values = [0.9455492842274187, 0.9466575225083778, 0.9508753468110867, 0.9502785577400493, 0.9442578358174003, 0.9582433020050865, 0.9576227704262268, 0.9476258285651071, 0.9566724924996812, 0.9476969459260068]

# 假设你有这样的数据 15mm
# mae_values = [0.017284608, 0.017413503, 0.018427745, 0.018379416, 0.017352575, 0.018341295, 0.017328398, 0.018304028, 0.018031027, 0.017178822]
# mse_values = [0.00052819337, 0.00052280416, 0.000613482, 0.0005956763, 0.0005413781, 0.00059447973, 0.0005427435, 0.00056393305, 0.00057639607, 0.00052972825]
# rmse_values = [0.022982458, 0.022864912, 0.024768569, 0.024406482, 0.023267534, 0.024381954, 0.023296855, 0.023747275, 0.02400825, 0.023015827]
# r2_values = [0.9326213081354561, 0.9333087790790451, 0.9217415099529563, 0.9240128817952293, 0.9309393989900588, 0.9241655206188074, 0.9307652265305845, 0.9280621934895185, 0.9264723519181376, 0.9325254872836113]


# 假设你有这样的数据 10mm
# mae_values = [0.02704904, 0.02688995, 0.027962448, 0.027388193, 0.026507894, 0.026796604, 0.026945813, 0.02693676, 0.028317217, 0.02696745]
# mse_values = [0.0018169429, 0.001783095, 0.0018775656, 0.0018127341, 0.0017877843, 0.0018061803, 0.0018038737, 0.0018411634, 0.0019024553, 0.0017484524]
# rmse_values = [0.042625614, 0.04222671, 0.043330885, 0.042576216, 0.042282198, 0.04249918, 0.042472035, 0.04290878, 0.043617144, 0.0418145]
# r2_values = [0.9579522798159749, 0.9587355894036145, 0.9565493462786324, 0.9580496792847144, 0.958627068430083, 0.9582013438432526, 0.9582547270264083, 0.9573917676323372, 0.9559733503788355, 0.9595372876079212]

# 假设你有这样的数据 5mm
# mae_values = [0.104361095, 0.101767786, 0.10368886, 0.104218155, 0.105652615, 0.10400997, 0.10952458, 0.100924365, 0.10291215, 0.107671745]
# mse_values = [0.031411353, 0.031216087, 0.031244084, 0.03152024, 0.031403773, 0.031106435, 0.03251418, 0.031725604, 0.03076978, 0.031110736]
# rmse_values = [0.17723249, 0.17668074, 0.17675996, 0.17753941, 0.17721109, 0.17637016, 0.1803169, 0.17811683, 0.17541318, 0.17638236]
# r2_values = [0.9493592940161799, 0.9496740981404409, 0.9496289607253967, 0.9491837567747562, 0.9493715169890279, 0.9498508782447367, 0.9475813429995984, 0.9488502034719193, 0.950391234220445, 0.9498415322869636]



# # x轴数据，即每次迭代的次数
# iterations = list(range(1, 11))

# # 画图
# plt.figure(figsize=(10, 5))

# plt.plot(iterations, mae_values, label='MAE')
# plt.plot(iterations, mse_values, label='MSE')
# plt.plot(iterations, rmse_values, label='RMSE')
# # plt.plot(iterations, r2_values, label='R2')

# plt.xlabel('Iterations')
# plt.ylabel('Metrics Value')
# plt.legend()
# plt.title('Ten-fold cross validation result for 5mm')
# plt.grid(True)
# plt.show()


# 假设你有这样的数据
# mm_values = ['20mm', '15mm', '10mm', '5mm']
# mae_values = [0.009012538, 0.013676002, 0.022519358, 0.10536194]
# mse_values = [0.00012531392, 0.00041162246, 0.0010839341, 0.03019165]
# rmse_values = [0.01119437, 0.02028848, 0.032923155, 0.17375745]
# r2_values = [0.939611008422347, 0.9554840443753447, 0.9741128520764973, 0.9501904112375013]

# # 画图
# plt.figure(figsize=(10, 5))

# plt.plot(mm_values, mae_values, label='MAE')
# plt.plot(mm_values, mse_values, label='MSE')
# plt.plot(mm_values, rmse_values, label='RMSE')
# plt.plot(mm_values, r2_values, label='R2')

# plt.xlabel('Grid Size')
# plt.ylabel('Metrics Value')
# plt.legend()
# plt.title('Metrics Change for Different Grid Sizes')
# plt.grid(True)
# plt.show()


# import numpy as np

# mae_values = [0.104361095, 0.101767786, 0.10368886, 0.104218155, 0.105652615, 0.10400997, 0.10952458, 0.100924365, 0.10291215, 0.107671745]
# mse_values = [0.031411353, 0.031216087, 0.031244084, 0.03152024, 0.031403773, 0.031106435, 0.03251418, 0.031725604, 0.03076978, 0.031110736]
# rmse_values = [0.17723249, 0.17668074, 0.17675996, 0.17753941, 0.17721109, 0.17637016, 0.1803169, 0.17811683, 0.17541318, 0.17638236]
# r2_values = [0.9493592940161799, 0.9496740981404409, 0.9496289607253967, 0.9491837567747562, 0.9493715169890279, 0.9498508782447367, 0.9475813429995984, 0.9488502034719193, 0.950391234220445, 0.9498415322869636]

# # 计算平均值
# average_mae = np.mean(mae_values)
# average_mse = np.mean(mse_values)
# average_rmse = np.mean(rmse_values)
# average_r2 = np.mean(r2_values)

# # 计算标准误差
# std_err_mae = np.std(mae_values, ddof=1) / np.sqrt(len(mae_values))
# std_err_mse = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))
# std_err_rmse = np.std(rmse_values, ddof=1) / np.sqrt(len(rmse_values))
# std_err_r2 = np.std(r2_values, ddof=1) / np.sqrt(len(r2_values))

# print(f"Average MAE: {average_mae}, Standard Error: {std_err_mae}")
# print(f"Average MSE: {average_mse}, Standard Error: {std_err_mse}")
# print(f"Average RMSE: {average_rmse}, Standard Error: {std_err_rmse}")
# print(f"Average R2: {average_r2}, Standard Error: {std_err_r2}")


# import numpy as np

# mae_values = [0.02704904, 0.02688995, 0.027962448, 0.027388193, 0.026507894, 0.026796604, 0.026945813, 0.02693676, 0.028317217, 0.02696745]
# mse_values = [0.0018169429, 0.001783095, 0.0018775656, 0.0018127341, 0.0017877843, 0.0018061803, 0.0018038737, 0.0018411634, 0.0019024553, 0.0017484524]
# rmse_values = [0.042625614, 0.04222671, 0.043330885, 0.042576216, 0.042282198, 0.04249918, 0.042472035, 0.04290878, 0.043617144, 0.0418145]
# r2_values = [0.9579522798159749, 0.9587355894036145, 0.9565493462786324, 0.9580496792847144, 0.958627068430083, 0.9582013438432526, 0.9582547270264083, 0.9573917676323372, 0.9559733503788355, 0.9595372876079212]

# # 计算平均值
# average_mae = np.mean(mae_values)
# average_mse = np.mean(mse_values)
# average_rmse = np.mean(rmse_values)
# average_r2 = np.mean(r2_values)

# # 计算标准误差
# std_err_mae = np.std(mae_values, ddof=1) / np.sqrt(len(mae_values))
# std_err_mse = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))
# std_err_rmse = np.std(rmse_values, ddof=1) / np.sqrt(len(rmse_values))
# std_err_r2 = np.std(r2_values, ddof=1) / np.sqrt(len(r2_values))

# print(f"Average MAE: {average_mae}, Standard Error: {std_err_mae}")
# print(f"Average MSE: {average_mse}, Standard Error: {std_err_mse}")
# print(f"Average RMSE: {average_rmse}, Standard Error: {std_err_rmse}")
# print(f"Average R2: {average_r2}, Standard Error: {std_err_r2}")


# import numpy as np

# mae_values = [0.017284608, 0.017413503, 0.018427745, 0.018379416, 0.017352575, 0.018341295, 0.017328398, 0.018304028, 0.018031027, 0.017178822]
# mse_values = [0.00052819337, 0.00052280416, 0.000613482, 0.0005956763, 0.0005413781, 0.00059447973, 0.0005427435, 0.00056393305, 0.00057639607, 0.00052972825]
# rmse_values = [0.022982458, 0.022864912, 0.024768569, 0.024406482, 0.023267534, 0.024381954, 0.023296855, 0.023747275, 0.02400825, 0.023015827]
# r2_values = [0.9326213081354561, 0.9333087790790451, 0.9217415099529563, 0.9240128817952293, 0.9309393989900588, 0.9241655206188074, 0.9307652265305845, 0.9280621934895185, 0.9264723519181376, 0.9325254872836113]

# # 计算平均值
# average_mae = np.mean(mae_values)
# average_mse = np.mean(mse_values)
# average_rmse = np.mean(rmse_values)
# average_r2 = np.mean(r2_values)

# # 计算标准误差
# std_err_mae = np.std(mae_values, ddof=1) / np.sqrt(len(mae_values))
# std_err_mse = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))
# std_err_rmse = np.std(rmse_values, ddof=1) / np.sqrt(len(rmse_values))
# std_err_r2 = np.std(r2_values, ddof=1) / np.sqrt(len(r2_values))

# print(f"Average MAE: {average_mae}, Standard Error: {std_err_mae}")
# print(f"Average MSE: {average_mse}, Standard Error: {std_err_mse}")
# print(f"Average RMSE: {average_rmse}, Standard Error: {std_err_rmse}")
# print(f"Average R2: {average_r2}, Standard Error: {std_err_r2}")


# import numpy as np

# mae_values = [0.007978464, 0.009161059, 0.00881325, 0.007934828, 0.008498118, 0.0076440196, 0.007022226, 0.008286227, 0.0076667094, 0.008117117]
# mse_values = [0.000133446, 0.00013073, 0.00012039307, 0.00012185566, 0.00013661105, 0.00010233594, 0.000103856706, 0.00012835687, 0.000106185624, 0.00012818261]
# rmse_values = [0.011551883, 0.0114337215, 0.010972378, 0.011038825, 0.011688073, 0.010116123, 0.010191011, 0.011329469, 0.010304641, 0.011321776]
# r2_values = [0.9455492842274187, 0.9466575225083778, 0.9508753468110867, 0.9502785577400493, 0.9442578358174003, 0.9582433020050865, 0.9576227704262268, 0.9476258285651071, 0.9566724924996812, 0.9476969459260068]

# # 计算平均值
# average_mae = np.mean(mae_values)
# average_mse = np.mean(mse_values)
# average_rmse = np.mean(rmse_values)
# average_r2 = np.mean(r2_values)

# # 计算标准误差
# std_err_mae = np.std(mae_values, ddof=1) / np.sqrt(len(mae_values))
# std_err_mse = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))
# std_err_rmse = np.std(rmse_values, ddof=1) / np.sqrt(len(rmse_values))
# std_err_r2 = np.std(r2_values, ddof=1) / np.sqrt(len(r2_values))

# print(f"Average MAE: {average_mae}, Standard Error: {std_err_mae}")
# print(f"Average MSE: {average_mse}, Standard Error: {std_err_mse}")
# print(f"Average RMSE: {average_rmse}, Standard Error: {std_err_rmse}")
# print(f"Average R2: {average_r2}, Standard Error: {std_err_r2}")


# 5mm Data
# models = ['LSTM', 'SVM', 'GRU', 'CNN']
# mae_values = [0.3067, 0.4569, 0.2566, 0.1053]
# mse_values = [0.1580, 0.3637, 0.1544, 0.0301]
# rmse_values = [0.3963, 0.6005, 0.3930, 0.1737]
# r2_values = [0.9140, 0.8036, 0.9228, 0.9501]

# 10mm Data
# models = ['LSTM', 'SVM', 'GRU', 'CNN']
# mae_values = [0.3002, 0.3592, 0.2573, 0.0225]
# mse_values = [0.1596, 0.2206, 0.1476, 0.0010]
# rmse_values = [0.3981, 0.4688, 0.3842, 0.0329]
# r2_values = [0.8921, 0.8518, 0.9123, 0.9741]


# 15mm Data
# models = ['LSTM', 'SVM', 'GRU', 'CNN']
# mae_values = [0.3129, 0.3239, 0.2613, 0.0136]
# mse_values = [0.1664, 0.1784, 0.1455, 0.0004]
# rmse_values = [0.4071, 0.4215, 0.3814, 0.0202]
# r2_values = [0.8750, 0.8668, 0.9015, 0.9554]

# 20mm Data
# models = ['LSTM', 'SVM', 'GRU', 'CNN']
# mae_values = [0.3028, 0.2925, 0.2957, 0.0090]
# mse_values = [0.1640, 0.1452, 0.1747, 0.0001]
# rmse_values = [0.4043, 0.3796, 0.4179, 0.0111]
# r2_values = [0.8751, 0.8899, 0.8609, 0.9396]

# Plotting
# plt.figure(figsize=(10, 6))

# plt.plot(models, mae_values, marker='o', label='MAE')
# plt.plot(models, mse_values, marker='o', label='MSE')
# plt.plot(models, rmse_values, marker='o', label='RMSE')
# plt.plot(models, r2_values, marker='o', label='R2')

# plt.title('Model Performance for 5mm', fontsize=26)
# plt.xlabel('Models', fontsize=24)
# plt.ylabel('Performance Metrics', fontsize=24)
# plt.legend(fontsize=22)
# plt.grid(True)
# plt.show()


import matplotlib.pyplot as plt

# Data
thickness = ['5mm', '10mm', '15mm', '20mm']
#MAE
# lstm_values = [0.3067, 0.3002, 0.3129, 0.3028]
# svm_values = [0.4569, 0.3592, 0.3239, 0.2925]
# gru_values = [0.2566, 0.2573, 0.2613, 0.2957]
# cnn_values = [0.1053, 0.0225, 0.0136, 0.0090]
#MSE
# lstm_values = [0.1580, 0.1596, 0.1664, 0.1640]
# svm_values = [0.3637, 0.2206, 0.1784, 0.1452]
# gru_values = [0.1544, 0.1476, 0.1455, 0.1747]
# cnn_values = [0.0301, 0.0010, 0.0004, 0.0001]
#RMSE
# lstm_values = [0.3963, 0.3981, 0.4071, 0.4043]
# svm_values = [0.6005, 0.4688, 0.4215, 0.3796]
# gru_values = [0.3930, 0.3842, 0.3814, 0.4179]
# cnn_values = [0.1737, 0.0329, 0.0202, 0.0111]

#R2
lstm_values = [0.9140, 0.8921, 0.8750, 0.8751]
svm_values = [0.8036, 0.8518, 0.8668, 0.8899]
gru_values = [0.9228, 0.9123, 0.9015, 0.8609]
cnn_values = [0.9501, 0.9741, 0.9554, 0.9396]


# Plotting
plt.figure(figsize=(12, 8))

plt.plot(thickness, lstm_values, marker='o', label='LSTM')
plt.plot(thickness, svm_values, marker='o', label='SVM')
plt.plot(thickness, gru_values, marker='o', label='GRU')
plt.plot(thickness, cnn_values, marker='o', label='CNN')

plt.title('Model Performance for Different Thicknesses', fontsize=26)
plt.xlabel('Material Thickness', fontsize=24)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=24)
plt.legend(fontsize=22)
plt.grid(True)
plt.show()
