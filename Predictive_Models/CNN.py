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
def create_sequence_data(data, sequence_length):
    X, Y = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:(i + sequence_length)].values
        label = data.iloc[i + sequence_length]
        X.append(seq)
        Y.append(label)
    return np.array(X), np.array(Y)

sequence_length = 10  # Set the sequence length
trainX, trainY = create_sequence_data(train['flow'], sequence_length)
testX, testY = create_sequence_data(test['flow'], sequence_length)

# Reshape data for CNN input
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=optimizer, loss=Huber())
# Apply early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(trainX, trainY, epochs=1000, batch_size=64, validation_data=(testX, testY), verbose=2, callbacks=[early_stopping])

# Predictions for training and testing sets
train_predictions = model.predict(trainX)
test_predictions = model.predict(testX)

# Inverse normalization to get original flow values
trainPredict = scaler.inverse_transform(train_predictions)
testPredict = scaler.inverse_transform(test_predictions)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testY = scaler.inverse_transform(testY.reshape(-1, 1))

# Calculate RMSE
trainScore= np.sqrt(mean_squared_error(trainY, trainPredict))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Train RMSE: %.2f' % (trainScore))
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
fig, ax = plt.subplots(figsize=(65, 20))
ax.plot(train.index[:len(trainPredict)], trainY, label='Actual', linewidth=1.5)
ax.plot(train.index[:len(trainPredict)], trainPredict, label='Predicted', linewidth=1.5)
format_xaxis_12h(ax, train)
plt.title('Train Set: Actual vs Predicted', fontsize=35)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Flow', fontsize=30)
plt.legend(fontsize=30)
plt.grid(True)
ax.tick_params(axis='x', labelsize=30)
plt.yticks(fontsize=30)
plt.savefig('results/Train_Test_Prediction_Graphics/CNN_Train_Set_Actual_vs_Predicted.png')
plt.show()

# Plot actual vs predicted values for the test set
fig, ax = plt.subplots(figsize=(65, 20))
ax.plot(test.index[:len(testPredict)], testY, label='Actual', linewidth=1.5)
ax.plot(test.index[:len(testPredict)], testPredict, label='Predicted',  linewidth=1.5)
format_xaxis_12h(ax, test)
plt.title('Test Set: Actual vs Predicted', fontsize=35)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Flow', fontsize=30)
plt.grid(True)
plt.legend(fontsize=30)
ax.tick_params(axis='x', labelsize=30)
plt.yticks(fontsize=30)
plt.savefig('results/Train_Test_Prediction_Graphics/CNN_Test_Set_Actual_vs_Predicted.png')
plt.show()
testPredict = testPredict.flatten()
testY = testY.flatten()

# get length
length = min(len(testPredict), len(testY), len(test.index))

results = pd.DataFrame({
    'datetime': test.index[:length],
    'actual': testY[:length],
    'predicted': testPredict[:length]
})


# output
results.to_csv('results/Train_Test_Prediction_data/CNN_Test_Set_Predictions.csv')

print("Test set predictions saved to 'results/Train_Test_Prediction_data/CNN_Test_Set_Predictions.csv'")
