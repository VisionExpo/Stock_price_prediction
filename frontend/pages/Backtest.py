import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(page_title="Backtesting Engine", page_icon="⚙️", layout="wide")
st.title("⚙️ Model Backtesting Engine")

BASE_API_URL = "http://backend:8000"

st.info("This page simulates a simple trading strategy using the model's historical predictions to see how it would have performed.")

# --- Inputs ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", "AAPL").upper()
with col2:
    start_date = st.date_input("Start Date", date(2023, 1, 1))
with col3:
    end_date = st.date_input("End Date", date(2024, 1, 1))

# --- Backtest Logic ---
if st.button("Run Backtest"):
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
    else:
        with st.spinner("Running backtest simulation... This may take a while."):
            try:
                payload = {
                    "ticker": ticker,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
                
                response = requests.post(f"{BASE_API_URL}/backtest", json=payload)
                response.raise_for_status()
                
                results = response.json()
                
                # --- Display Metrics ---
                st.subheader("Backtest Results")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(
                        label="Total Return",
                        value=f"{results['total_return_pct']:.2f}%",
                        delta=f"{results['total_return_pct']:.2f}%"
                    )
                with metric_col2:
                    st.metric(
                        label="Final Portfolio Value",
                        value=f"${results['final_portfolio_value']:,.2f}",
                        delta=f"${results['final_portfolio_value'] - 10000:,.2f}"
                    )
                
                # --- Display Equity Curve Chart ---
                st.subheader("Portfolio Growth (Equity Curve)")
                equity_curve_df = pd.DataFrame(results['equity_curve'])
                equity_curve_df['Date'] = pd.to_datetime(equity_curve_df['Date'])
                equity_curve_df = equity_curve_df.set_index('Date')
                
                st.line_chart(equity_curve_df)

            except requests.exceptions.RequestException as e:
                if e.response:
                    error_details = e.response.json().get('detail', 'An unknown error occurred.')
                    st.error(f"Backtest failed (Code {e.response.status_code}): {error_details}")
                else:
                    st.error(f"Connection Error: Could not connect to the backend API.")