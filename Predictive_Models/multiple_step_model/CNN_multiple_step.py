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
import math
# Load the data
data1 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_1.csv', parse_dates=['datetime'], index_col='datetime')
data2 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_2.csv', parse_dates=['datetime'], index_col='datetime')
data3 = pd.read_csv('../data/Processed/K1D43/K1D43_data_period_3.csv', parse_dates=['datetime'], index_col='datetime')

# Combine data1 and data2 to create the training dataset
train = pd.concat([data1, data2])
test = data3

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform the 'flow' column
train['flow'] = scaler.fit_transform(train[['flow']])

# Transform the 'flow' column of the test data using the fitted scaler
test['flow'] = scaler.transform(test[['flow']])

# Function to create sequence data
def create_multi_step_sequence_data(data, sequence_length, steps_ahead):
    X, Y = [], []
    for i in range(len(data) - sequence_length - steps_ahead + 1):
        seq = data.iloc[i:(i + sequence_length)].values
        label = data.iloc[i + sequence_length:i + sequence_length + steps_ahead].values
        X.append(seq)
        Y.append(label)
    return np.array(X), np.array(Y)

def train_predict_evaluate(steps_ahead, train, test, sequence_length=10, epochs=1000, batch_size=64):
    # dataset
    trainX, trainY = create_multi_step_sequence_data(train['flow'], sequence_length, steps_ahead)
    testX, testY = create_multi_step_sequence_data(test['flow'], sequence_length, steps_ahead)

    # reshape dataset
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # build model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(steps_ahead))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
    # Apply early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(trainX, trainY, epochs=1000, batch_size=64, validation_data=(testX, testY), verbose=2, callbacks=[early_stopping])

    # Prediction
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # inverse normalization
    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1)).reshape(-1, steps_ahead)
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1)).reshape(-1, steps_ahead)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1)).reshape(-1, steps_ahead)
    testY = scaler.inverse_transform(testY.reshape(-1, 1)).reshape(-1, steps_ahead)

    # RMSE
    trainScore = np.sqrt(mean_squared_error(trainY.reshape(-1), trainPredict.reshape(-1)))
    testScore = np.sqrt(mean_squared_error(testY.reshape(-1), testPredict.reshape(-1)))

    return trainScore, testScore


# different steps RMSE
train_rmse_3, test_rmse_3 = train_predict_evaluate(3, train, test)
train_rmse_6, test_rmse_6 = train_predict_evaluate(6, train, test)
train_rmse_12, test_rmse_12 = train_predict_evaluate(12, train, test)

# print
print(f'3 Steps - Train RMSE: {train_rmse_3}, Test RMSE: {test_rmse_3}')
print(f'6 Steps - Train RMSE: {train_rmse_6}, Test RMSE: {test_rmse_6}')
print(f'12 Steps - Train RMSE: {train_rmse_12}, Test RMSE: {test_rmse_12}')
# Function to format the x-axis with 12-hour intervals

