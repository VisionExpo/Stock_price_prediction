import streamlit as st
import requests
import pandas as pd
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# --- App Title ---
st.title("📈 AI Stock Price Predictor")
st.caption("This app uses an LSTM model with sentiment analysis to predict the next day's closing price.")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    # Ticker input
    ticker = st.text_input("Enter a Stock Ticker", "AAPL").upper()

with col2:
    # Date input
    selected_date = st.date_input("Select a Date", date(2024, 9, 26))

# --- Prediction Logic ---
if st.button("Predict"):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner('Fetching prediction from the AI model...'):
            # Define the API endpoint and the request payload
            api_url = "http://backend:8000/predict"
            payload = {
                "ticker": ticker,
                "date": selected_date.strftime("%Y-%m-%d")
            }
            
            try:
                # Make the POST request to the backend API
                response = requests.post(api_url, json=payload)
                
                # Check the response from the API
                if response.status_code == 200:
                    prediction = response.json()
                    predicted_price = prediction.get('predicted_price')
                    st.success(f"**Predicted Close Price for {ticker}**: ${predicted_price:.2f}")
                else:
                    error_details = response.json().get('detail', 'An unknown error occurred.')
                    st.error(f"Error from API (Code {response.status_code}): {error_details}")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the backend API. Please ensure it's running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")