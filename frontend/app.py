import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# --- App Title ---
st.title("📈 AI Stock Price Predictor")
st.caption("This app uses an LSTM model with sentiment analysis to predict the next day's closing price.")

# --- API URLs ---
# Use the Docker service name 'backend'
BASE_API_URL = "http://backend:8000"

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter a Stock Ticker", "AAPL").upper()
with col2:
    # Default to yesterday for prediction
    yesterday = date.today() - timedelta(days=1)
    selected_date = st.date_input("Select a Date for Prediction", yesterday)

# --- Prediction Logic ---
if st.button("Predict"):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner('Working our magic... ✨'):
            # --- Call History Endpoint ---
            try:
                history_response = requests.get(f"{BASE_API_URL}/history/{ticker}")
                history_response.raise_for_status()
                history_data = history_response.json()
                
                history_df = pd.DataFrame(history_data)
                history_df['Date'] = pd.to_datetime(history_df['Date'])
                history_df = history_df.set_index('Date')

                # Display the chart of the last 90 days
                st.subheader(f"Recent Price History for {ticker}")
                st.line_chart(history_df['Close'].tail(90))

            except requests.exceptions.RequestException as e:
                st.error(f"Could not fetch historical data: {e}")

            # --- Call Prediction Endpoint ---
            try:
                predict_payload = {"ticker": ticker, "date": selected_date.strftime("%Y-%m-%d")}
                predict_response = requests.post(f"{BASE_API_URL}/predict", json=predict_payload)
                predict_response.raise_for_status()
                
                prediction = predict_response.json()
                predicted_price = prediction.get('predicted_price')
                
                st.success(f"**Predicted Close Price for {ticker} on {selected_date + timedelta(days=1)}**: ${predicted_price:.2f}")

            except requests.exceptions.RequestException as e:
                error_details = e.response.json().get('detail', 'An unknown error occurred.')
                st.error(f"Prediction failed (Code {e.response.status_code}): {error_details}")