import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Prepare data for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)
# Predict future prices
def predict_strike_price(stock, days_to_predict):
    df_close = stock['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

    time_step = 45
    X, y = create_dataset(df_close, time_step)

    X = X.reshape(X.shape[0], X.shape[1], 1)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def predict_strike_price(stock, days_to_predict):
    if 'Close' not in stock.columns:
        raise ValueError("Stock data is missing the 'Close' column.")

    df_close = stock['Close'].dropna()

    if df_close.empty:
        raise ValueError("The 'Close' column has no data after dropping NaNs.")

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close_scaled = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

    # Create dataset
    time_step = 45
    X, y = [], []

    for i in range(time_step, len(df_close_scaled)):
        X.append(df_close_scaled[i - time_step:i, 0])
        y.append(df_close_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create training samples.")

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=0)

    # Predict future
    input_seq = df_close_scaled[-time_step:]
    input_seq = input_seq.reshape(1, -1, 1)

    predictions = []

    for _ in range(days_to_predict):
        next_pred = model.predict(input_seq, verbose=0)
        predictions.append(next_pred[0][0])
        input_seq = np.append(input_seq[:, 1:, :], [[[next_pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=64, verbose=0)

    X_input = df_close[-time_step:].reshape(1, time_step, 1)

    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0][0])
        X_input = np.append(X_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Streamlit UI
st.title("📈 Stock Price Predictor (LSTM)")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
days_to_predict = st.number_input("Days to Predict", min_value=1, max_value=30, value=5)

if st.button("Predict"):
    with st.spinner("Fetching data and making predictions..."):
        data = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if data.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            predictions = predict_strike_price(data, days_to_predict)
            st.success("Prediction complete!")
            st.write("📊 Predicted closing prices:")

            for i, price in enumerate(predictions, 1):
                st.write(f"{i} day(s) ahead: **${price[0]:.2f}**")



