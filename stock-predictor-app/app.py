import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor")

# User input for stock ticker
ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")

if ticker:
    # Download stock data
    df = yf.download(ticker, start="2015-01-01")

    if df.empty:
        st.error("⚠️ Couldn't download data for this ticker.")
    else:
        # Prepare data
        df = df[['Close']].dropna()
        df['Date'] = df.index
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

        # Train linear regression model
        X = df[['Date_ordinal']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)
        df['Predicted_Close'] = model.predict(X)

        # Safely plot if columns exist and are not empty
        if 'Close' in df.columns and 'Predicted_Close' in df.columns:
            cleaned_df = df[['Close', 'Predicted_Close']].dropna()
            if not cleaned_df.empty:
                st.line_chart(cleaned_df)
            else:
                st.error("⚠️ Chart can't be shown because data is empty after removing missing values.")
        else:
            st.error("⚠️ Columns 'Close' or 'Predicted_Close' not found.")

        # Show the most recent close price
        st.write("Most recent close:", f"${df['Close'].iloc[-1]:.2f}")

        # Preview last few rows
        st.subheader("📊 Data Preview")
        st.dataframe(df.tail())


