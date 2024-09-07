import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


# Set up the parameters
num_layers = 4
num_neurons = 512
dropout_rate = 0.02
prediction_days = 500

# Get the data
crypto_currency = 'BTC'
against_currency = 'USD'
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
data = yf.download(f'{crypto_currency}-{against_currency}', start, end)

# Prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create the model
model = Sequential()
for i in range(num_layers):
    model.add(LSTM(units=num_neurons, return_sequences=True))
    model.add(Dropout(dropout_rate))
model.add(Dense(units=1))

# Compile the model
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Prepare the training data
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Predict the next 30-60 days' prices
future_period = 60  # Adjust this for the desired number of days
predicted_prices = []

# Use the last prediction_days data as input for predicting future prices
last_data = scaled_data[-prediction_days:].reshape(1, -1, 1)

for _ in range(future_period):
    prediction = model.predict(last_data)[0][0]
    predicted_prices.append(prediction)
    last_data = np.append(last_data[:, 1:, :], [[prediction]], axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plotting the predicted prices
plt.plot(data.index[-prediction_days:], scaled_data[-prediction_days:], color='blue', label='Actual Prices')
plt.plot(data.index[-prediction_days:][:future_period], predicted_prices, color='red', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
