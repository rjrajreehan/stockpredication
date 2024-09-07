from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        prediction_days = 1


        # Load the data
        crypto_currency = 'BTC'
        against_currency = 'USD'
        start = dt.datetime(2015, 1, 1)
        end = dt.datetime.now()
        data = yf.download(f'{crypto_currency}-{against_currency}', start, end)

        # Prepare the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        model_inputs = scaled_data[-prediction_days:]
        x_test = []

        # Reshape the model inputs
        x_test = np.reshape(model_inputs, (1, prediction_days, 1))



        # Load the trained model
        model = load_model(r"D:\STUDY\AI DATA SCIENCE\Deep Learning\Predicting Future Trends of Crypto\my_model.h5")  # Replace with your model path

        # Make predictions
        prediction_prices = model.predict(x_test)
        prediction_prices = np.squeeze(prediction_prices)  # Remove the extra dimension
        prediction_prices = scaler.inverse_transform(prediction_prices.reshape(-1, 1))  # Reshape for inverse transform

        # Generate dates for the upcoming days
        last_date = data.index[-1]
        upcoming_dates = pd.date_range(last_date, periods=prediction_days + 1, closed='right')[1:]

        # Plot actual and predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], color='black', label='Actual Prices')
        plt.plot(upcoming_dates, prediction_prices, color='green', label='Predicted Prices')
        plt.title(f'{crypto_currency} price prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Save the plot to a file
        plot_path = 'static/plot.png'
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', plot_path=plot_path)

    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
