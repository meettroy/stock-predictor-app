import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📈 Stock Price Predictor (Linear Regression)")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")

if ticker:
    df = yf.download(ticker, start="2015-01-01", group_by='ticker')
    df = df[['Close']].copy()
    df.columns = ['Close']  # Flatten MultiIndex
    df['Date'] = df.index
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

    X = df[['Date_ordinal']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    df['Predicted_Close'] = model.predict(X)

    st.line_chart(df[['Close', 'Predicted_Close']])
    st.write("Most recent close:", df['Close'].iloc[-1])


