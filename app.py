import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def download_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

def create_dataset(dataset, time_step=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_step, 0])
    return np.array(data_X), np.array(data_Y)

def predict_strike_price(stock, days_to_predict):
    df_close = stock['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

    time_step = 45
    X, y = create_dataset(df_close, time_step)

    X_train, X_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_test = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, verbose=0)

    last_days_data = np.array(df_close[-time_step:]).reshape(1, -1)
    X_input = last_days_data.reshape((1, time_step, 1))

    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0][0])
        X_input = np.append(X_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit UI
st.title("📈 Stock Strike Price Predictor")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start date")
end_date = st.date_input("End date")
days_to_predict = st.number_input("How many future days to predict?", min_value=1, max_value=30, value=5)

if st.button("Predict"):
    with st.spinner("Downloading data and training model..."):
        try:
            stock_data = download_stock_data(ticker, str(start_date), str(end_date))
            if stock_data.empty:
                st.error("No data found for given ticker and dates.")
            else:
                predictions = predict_strike_price(stock_data, days_to_predict)
                st.success("Prediction complete!")
                for i, price in enumerate(predictions, start=1):
                    st.write(f"{i} day(s) ahead: **${price[0]:.2f}**")
        except Exception as e:
            st.error(f"Error: {str(e)}")

