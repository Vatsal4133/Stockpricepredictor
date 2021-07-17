import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# load data
Company = "FB"

# dataset dates
start = dt.datetime(2020, 6, 25)
end = dt.datetime(2021, 6, 10)

# data
data = web.DataReader(Company, 'yahoo', start, end)

# prepare dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0, 2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0, 2))
model.add(LSTM(units=50))
model.add(Dropout(0, 2))
model.add(Dense(units=1))  # prediction layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


'''Testing model accuracy'''

# Load Test Data
Test_start = dt.datetime(2021, 6, 10)
Test_end = dt.datetime.now()

Test_data = web.DataReader(Company, 'yahoo', Test_start, Test_end)
actual_prices = Test_data['Close'].values
total_dataset = pd.concat((data['Close'], Test_data['Close']), axis=0)
model_input = total_dataset[len(total_dataset)-len(Test_data)-prediction_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input)

# make prediction  on test data
x_test = []

for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the Test predictions
plt.plot(actual_prices, color='blue', label=f'Actual {Company} prices')
plt.plot(predicted_prices, color='red', label=f'Predicted {Company} prices')
plt.title(f'{Company} Share Prices')
plt.xlabel('time')
plt.ylabel(f'{Company} Share Prices')
plt.legend()
plt.show()
