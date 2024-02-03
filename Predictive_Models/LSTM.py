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
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# For example, use data from the past 3 timesteps
look_back = 3
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)

# Reshape into [samples, timesteps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define hyperparameters
lstm_units = 64  # Number of LSTM units
look_back = 3    # Timestep length
learning_rate = 0.002  # Learning rate
batch_size = 32 # Batch size
epochs = 100     # Number of training epochs
dropout_rate = 0.4  # Dropout rate

# Build LSTM network
model = Sequential()
model.add(LSTM(lstm_units, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units, return_sequences=False))
model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(trainX, trainY, validation_data=(testX, testY),
                    epochs=epochs, batch_size=batch_size,
                    verbose=2, callbacks=[early_stopping])

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse transform predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# Function to format the x-axis with 12-hour intervals
def format_xaxis_12h(ax, dataset):
    xticks = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq='12H')
    ax.set_xticks(xticks)
    ax.set_xticklabels([xt.strftime('%m-%d %H:%M') for xt in xticks], rotation=45, ha='right')

# Plot training and validation loss
plt.figure(figsize=(20, 10))
plt.plot(history.history['loss'], label='Train Loss', linewidth=1.2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=1.2)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
#plt.savefig('../Graphics/Model_Loss.png')
plt.show()


# Plot actual vs predicted values for the training set
fig, ax = plt.subplots(figsize=(65, 30))
ax.plot(train.index[:len(trainPredict)], trainY[0], label='Actual', linewidth=1.5)
ax.plot(train.index[:len(trainPredict)], trainPredict[:, 0], label='Predicted', linewidth=1.5)
format_xaxis_12h(ax, train)
plt.title('Train Set: Actual vs Predicted', fontsize=35)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Flow', fontsize=30)
plt.legend(fontsize=30)
plt.grid(True)
ax.tick_params(axis='x', labelsize=30)
plt.yticks(fontsize=30)
plt.savefig('results/Train_Test_Prediction_Graphics/LSTM_Train_Set_Actual_vs_Predicted.png')
plt.show()

# Plot actual vs predicted values for the test set
fig, ax = plt.subplots(figsize=(65, 30))
ax.plot(test.index[:len(testPredict)], testY[0], label='Actual', linewidth=1.5)
ax.plot(test.index[:len(testPredict)], testPredict[:, 0], label='Predicted', linewidth=1.5)
format_xaxis_12h(ax, test)
plt.title('Test Set: Actual vs Predicted', fontsize=35)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Flow', fontsize=30)
plt.legend(fontsize=30)
plt.grid(True)
ax.tick_params(axis='x', labelsize=30)
plt.yticks(fontsize=30)
plt.savefig('results/Train_Test_Prediction_Graphics/LSTM_Test_Set_Actual_vs_Predicted.png')
plt.show()

# new DataFrame
results = pd.DataFrame({
    'datetime': test.index[:len(testPredict)],
    'actual': testY[0],
    'predicted': testPredict[:, 0]
})

results.set_index('datetime', inplace=True)

# output
results.to_csv('results/Train_Test_Prediction_data/LSTM_Test_Set_Predictions.csv')

print("Test set predictions saved to 'results/Train_Test_Prediction_data/LSTM_Test_Set_Predictions.csv'")
