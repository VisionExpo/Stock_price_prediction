import streamlit as st
import requests
import pandas as pd
from datetime import date, timedelta

# --- API URLs ---
BASE_API_URL = "http://backend:8000"

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def get_available_tickers():
    try:
        response = requests.get(f"{BASE_API_URL}/tickers")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch ticker list: {e}")
        return []

def color_change(val):
    """Applies color to the 'Predicted Change (%)' column."""
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}'

# --- Screener Page ---
st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")

st.title("🔍 Stock Screener")

available_tickers = get_available_tickers()
if not available_tickers:
    st.warning("Could not load tickers from the API.")
else:
    selected_tickers = st.multiselect("Select stocks to screen", options=available_tickers, default=available_tickers[:4])
    
    prediction_date = date(2025, 9, 26)
    st.info(f"The screener will predict the closing price for **{prediction_date + timedelta(days=1)}** based on data up to **{prediction_date}**.")

    if st.button("Run Screener"):
        if not selected_tickers:
            st.error("Please select at least one ticker.")
        else:
            results = []
            placeholder = st.empty()
            progress_bar = st.progress(0, text="Starting Screener...")
            
            for i, ticker in enumerate(selected_tickers):
                progress_bar.progress((i) / len(selected_tickers), text=f"Screening {ticker}...")
                try:
                    predict_payload = {"ticker": ticker, "date": prediction_date.strftime("%Y-%m-%d")}
                    predict_response = requests.post(f"{BASE_API_URL}/predict", json=predict_payload)
                    
                    if predict_response.status_code == 200:
                        predicted_price = predict_response.json().get('predicted_price')
                        
                        history_response = requests.get(f"{BASE_API_URL}/history/{ticker}")
                        history_response.raise_for_status()
                        history_data = history_response.json()
                        last_price = history_data[-1]['Close'] if history_data else 0
                        
                        change_pct = ((predicted_price - last_price) / last_price) * 100 if last_price > 0 else 0
                        
                        results.append({
                            "Ticker": ticker, "Last Close": last_price, "Predicted Close": predicted_price, "Predicted Change (%)": change_pct
                        })
                    else:
                        results.append({"Ticker": ticker, "Last Close": "N/A", "Predicted Close": "Error", "Predicted Change (%)": 0})
                
                except requests.exceptions.RequestException:
                    results.append({"Ticker": ticker, "Last Close": "N/A", "Predicted Close": "Error", "Predicted Change (%)": 0})

                if results:
                    results_df = pd.DataFrame(results).sort_values(by="Predicted Change (%)", ascending=False)
                    placeholder.dataframe(
                        results_df.style.format({
                            "Last Close": "${:.2f}",
                            "Predicted Close": "${:.2f}",
                            "Predicted Change (%)": "{:.2f}%"
                        }).applymap(color_change, subset=['Predicted Change (%)']),
                        use_container_width=True
                    )

            progress_bar.progress(1.0, text="Screener finished!")