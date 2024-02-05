import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import numpy as np

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

# Function to create a supervised learning format dataset
def create_dataset(dataset, look_back, steps_ahead):
    X, Y = [], []
    for i in range(len(dataset) - look_back - steps_ahead):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:i + look_back + steps_ahead, 0])
    return np.array(X), np.array(Y)

# For example, use data from the past 3 timesteps
look_back = 3
steps_ahead_3 = 3
steps_ahead_6 = 6
steps_ahead_12 = 12


# Define hyperparameters
lstm_units = 64  # Number of LSTM units
look_back = 3    # Timestep length
learning_rate = 0.002  # Learning rate
batch_size = 32 # Batch size
epochs = 100     # Number of training epochs
dropout_rate = 0.4  # Dropout rate
def train_and_evaluate(look_back, steps_ahead):
    # dataset
    trainX, trainY = create_dataset(train.values, look_back, steps_ahead)
    testX, testY = create_dataset(test.values, look_back, steps_ahead)

    # reshape [samples, timesteps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # build LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(look_back, 1), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(steps_ahead))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(trainX, trainY, validation_data=(testX, testY),
                        epochs=epochs, batch_size=batch_size,
                        verbose=2, callbacks=[early_stopping])
    # Prediction
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # inverse normalization
    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1)).reshape(-1, steps_ahead)
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1)).reshape(-1, steps_ahead)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1)).reshape(-1, steps_ahead)
    testY = scaler.inverse_transform(testY.reshape(-1, 1)).reshape(-1, steps_ahead)

    # RMSE
    trainScore = math.sqrt(mean_squared_error(trainY.reshape(-1), trainPredict.reshape(-1)))
    testScore = math.sqrt(mean_squared_error(testY.reshape(-1), testPredict.reshape(-1)))
    return trainScore, testScore

train_rmse_3, test_rmse_3 = train_and_evaluate(look_back, 3)
train_rmse_6, test_rmse_6 = train_and_evaluate(look_back, 6)
train_rmse_12, test_rmse_12 = train_and_evaluate(look_back, 12)

print(f"3 steps - Train RMSE: {train_rmse_3}, Test RMSE: {test_rmse_3}")
print(f"6 steps - Train RMSE: {train_rmse_6}, Test RMSE: {test_rmse_6}")
print(f"12 steps - Train RMSE: {train_rmse_12}, Test RMSE: {test_rmse_12}")





