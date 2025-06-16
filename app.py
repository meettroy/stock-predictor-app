import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to prepare data and make predictions
def predict_strike_price(df, days_to_predict):
    df_close = df[['Close']]

    # Normalize
    scaler = MinMaxScaler()
    df_close_scaled = scaler.fit_transform(df_close)

    time_step = 60
    X, y = [], []

    for i in range(time_step, len(df_close_scaled)):
        X.append(df_close_scaled[i-time_step:i, 0])
        y.append(df_close_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into train/test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=0)

    # Predict future prices
    temp_input = df_close_scaled[-time_step:].reshape(1, time_step, 1)
    predictions = []

    for _ in range(days_to_predict):
        pred = model.predict(temp_input, verbose=0)[0][0]
        predictions.append(pred)
        temp_input = np.append(temp_input[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit UI
st.title("📈 Stock Price Predictor with LSTM")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")
start_date = st.date_input("Start date:", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date:", pd.to_datetime("today"))
days = st.slider("Days to predict:", 1, 30, 5)

if st.button("Predict"):
    with st.spinner("Downloading data and training model..."):
        df = download_stock_data(ticker, start_date, end_date)
        if df.empty or 'Close' not in df.columns:
            st.error("⚠️ Could not download data or 'Close' column missing.")
        else:
            predictions = predict_strike_price(df, days)
            st.success("✅ Prediction complete!")

            for i, price in enumerate(predictions):
                st.write(f"Day {i+1}: **${price[0]:.2f}**")

            # Debug info
            st.write("📊 Prediction preview:", predictions[:5])


