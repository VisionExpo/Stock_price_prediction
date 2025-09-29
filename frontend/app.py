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
BASE_API_URL = "http://backend:8000"

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter a Stock Ticker", "AAPL").upper()
with col2:
    default_date = date(2025, 9, 26) 
    selected_date = st.date_input("Select a Date for Prediction", default_date)

# --- Prediction Logic ---
if st.button("Predict"):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner('Working...'):
            predicted_price = None
            # --- Call Prediction Endpoint First ---
            try:
                predict_payload = {"ticker": ticker, "date": selected_date.strftime("%Y-%m-%d")}
                predict_response = requests.post(f"{BASE_API_URL}/predict", json=predict_payload)
                
                if predict_response.status_code == 200:
                    prediction = predict_response.json()
                    predicted_price = prediction.get('predicted_price')
                    st.success(f"**Predicted Close Price for {ticker} on {selected_date + timedelta(days=1)}**: ${predicted_price:.2f}")
                else:
                    error_details = predict_response.json().get('detail', 'An unknown prediction error occurred.')
                    st.error(f"Prediction failed (Code {predict_response.status_code}): {error_details}")

            except requests.exceptions.RequestException as e:
                st.error("Connection Error: Could not connect to the backend API for prediction.")

            # --- Call History Endpoint and Plot ---
            try:
                history_response = requests.get(f"{BASE_API_URL}/history/{ticker}")
                history_response.raise_for_status() # This will raise an exception for 4xx/5xx errors
                history_data = history_response.json()
                
                if not history_data:
                    st.warning("No historical data found for this ticker to plot.")
                else:
                    history_df = pd.DataFrame(history_data)
                    history_df['Date'] = pd.to_datetime(history_df['Date'])
                    history_df = history_df.set_index('Date')['Close'].tail(90)

                    if predicted_price is not None:
                        prediction_date = selected_date + timedelta(days=1)
                        prediction_series = pd.Series({pd.to_datetime(prediction_date): predicted_price})
                        history_df = pd.concat([history_df, prediction_series])
                        st.subheader(f"Price History and Forecast for {ticker}")
                    else:
                        st.subheader(f"Recent Price History for {ticker}")

                    st.line_chart(history_df)

            # --- UPDATED: More detailed error handling for the history endpoint ---
            except requests.exceptions.RequestException as e:
                if e.response is not None:
                    error_details = e.response.json().get('detail', 'Could not fetch historical data.')
                    st.warning(f"Chart Error (Code {e.response.status_code}): {error_details}")
                else:
                    st.warning("Connection Error: Could not connect to the backend API for historical data.")