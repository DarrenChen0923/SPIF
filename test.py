import numpy as np

# 假设你有一个包含这些值的列表
mae_values = [0.106506124, 0.09461953, 0.09531141, 0.09971751, 0.1037572, 0.098550335, 0.091872625, 0.0927017, 0.10436982, 0.112302504]
mse_values = [0.030022351, 0.02312919, 0.023940131, 0.02774053, 0.028744897, 0.024334501, 0.022207845, 0.028143872, 0.028570313, 0.033501264]
rmse_values = [0.17326958, 0.15208283, 0.15472598, 0.16655488, 0.1695432, 0.15599519, 0.14902297, 0.16776136, 0.16902755, 0.18303351]
r2_values = [0.9504697175654648, 0.9636786072707704, 0.9634772278221266, 0.9562033697569521, 0.9531702177270072, 0.9608564116264424, 0.9663180188756733, 0.9540273852704102, 0.955273662957146, 0.9445410794383151]

# 使用 NumPy 计算平均值和标准差
mae_mean = np.mean(mae_values)
mae_std = np.std(mae_values)

mse_mean = np.mean(mse_values)
mse_std = np.std(mse_values)

rmse_mean = np.mean(rmse_values)
rmse_std = np.std(rmse_values)

r2_mean = np.mean(r2_values)
r2_std = np.std(r2_values)

# 打印结果
print(f'MAE Mean: {mae_mean}, Standard Deviation: {mae_std}')
print(f'MSE Mean: {mse_mean}, Standard Deviation: {mse_std}')
print(f'RMSE Mean: {rmse_mean}, Standard Deviation: {rmse_std}')
print(f'R2 Mean: {r2_mean}, Standard Deviation: {r2_std}')
