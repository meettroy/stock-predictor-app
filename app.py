import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("📈 Stock Price Predictor")

# Input for ticker
ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")

if ticker:
    df = yf.download(ticker, start="2015-01-01")

    if df.empty:
        st.error("⚠️ Couldn't retrieve stock data. Please try a different ticker.")
    else:
        # Prepare data
        df = df[['Close']].dropna()
        df['Date'] = df.index
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

        # Train model
        X = df[['Date_ordinal']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)

        # Predict
        df['Predicted_Close'] = model.predict(X)

        # Chart
        if {'Close', 'Predicted_Close'}.issubset(df.columns):
            st.line_chart(df[['Close', 'Predicted_Close']])
        else:
            st.error("⚠️ Columns 'Close' and/or 'Predicted_Close' are missing from the DataFrame.")

        # Show latest data and preview
        st.write("Most recent close:", df['Close'].iloc[-1])
        st.write("Data preview:", df.tail())



