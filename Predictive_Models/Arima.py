import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.losses import Huber

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Load the data
data1 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_1.csv', parse_dates=['datetime'], index_col='datetime')
data2 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_2.csv', parse_dates=['datetime'], index_col='datetime')
data3 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_3.csv', parse_dates=['datetime'], index_col='datetime')

# Combine data1 and data2 to create the training dataset
train = pd.concat([data1, data2])
test = data3

auto_model = pm.auto_arima(train['flow'], seasonal=False, m=1,
                           d=None, max_p=5, max_q=5, trace=True,
                           error_action='ignore', suppress_warnings=True)

print(auto_model.summary())

# 使用确定的参数训练 ARIMA 模型
model = ARIMA(train['flow'], order=auto_model.order, seasonal_order=auto_model.seasonal_order)
model_fit = model.fit()

# summary of model
print(model_fit.summary())

# predict

predictions = model_fit.forecast(len(test))
predictions.index = test.index

#  RMSE
rmse = np.sqrt(mean_squared_error(test['flow'], predictions))
print(f'Test RMSE: {rmse}')

plt.figure(figsize=(45, 16))
plt.plot(test['flow'], label='Actual',linewidth=1.2)
plt.plot(predictions, label='Predicted', color='red',linewidth=1.2)
plt.legend(fontsize=30)
plt.title('ARIMA Flow Prediction', fontsize = 40)
plt.xlabel("Time", fontsize=35)
plt.ylabel("Value", fontsize=35)
plt.xticks(rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True)  #
plt.tight_layout()
plt.savefig('results/Train_Test_Prediction_Graphics/Arima_Test_Set.png')
plt.show()
plt.close()