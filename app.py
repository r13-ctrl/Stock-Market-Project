import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.title("📈 Live Stock Price Prediction App")

# User input
ticker = st.text_input("Enter Stock Symbol (Example: AAPL or RELIANCE.NS)", "AAPL")

# Load data
data = yf.download(ticker, period="10y")
data = data[['Open','High','Low','Close','Volume']]

# Feature engineering
data['Prev_Close'] = data['Close'].shift(1)
data['Prev_Open'] = data['Open'].shift(1)
data['Prev_High'] = data['High'].shift(1)
data['Prev_Low'] = data['Low'].shift(1)
data['Prev_Volume'] = data['Volume'].shift(1)
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Target'] = data['Close']
data = data.dropna()
if data.empty:
    st.error("No data found for this stock symbol. Please try another symbol like AAPL or RELIANCE.NS")
    st.stop()

# Features
X = data[['Prev_Open','Prev_High','Prev_Low','Prev_Close','Prev_Volume','MA10','MA50']]
y = data['Target']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_scaled, y)

# Predict next day price
latest = X_scaled[-1].reshape(1,-1)
prediction = model.predict(latest)

st.subheader("Predicted Next Day Closing Price:")
st.success(f"${prediction[0]:.2f}")

# Show chart
st.subheader("Historical Closing Price")
st.line_chart(data['Close'])