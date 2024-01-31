import optuna
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import EarlyStopping

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

# Create dataset in supervised learning format
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3  # For example, use data from the past 3 time steps
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)

# Reshape to [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

def objective(trial):
    # Define the hyperparameter search space
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(look_back, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')

    # Train model
    model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0)

    # Return validation loss
    val_loss = model.evaluate(testX, testY, verbose=0)
    return val_loss

# Create a study and optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Output best hyperparameters
best_hps = study.best_trial.params
print('Best trial:', best_hps)