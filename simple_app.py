"""
Simplified Streamlit web application for Stock Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from dotenv import load_dotenv
import pandas_datareader as pdr

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Create directories if they don't exist
data_path = "data"
model_path = "models"
results_path = "results"

for directory in [data_path, model_path, results_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_stock_data_tiingo(symbol):
    """
    Fetch stock data from Tiingo API
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv("TIINGO_API_KEY")
        if not api_key:
            st.error("TIINGO_API_KEY not found in environment variables. Please add it to .env file.")
            return None
        
        # Fetch data from Tiingo
        df = pdr.get_data_tiingo(symbol, api_key=api_key)
        
        # Save data to CSV
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        df.to_csv(os.path.join(data_path, f'{symbol}.csv'))
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def main():
    """
    Main function for the Streamlit app
    """
    st.title("Stock Price Prediction App")
    st.write("Predict future stock prices using LSTM neural networks")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Viewer", "About"])
    
    if page == "Data Viewer":
        st.header("Stock Data Viewer")
        
        symbol = st.text_input("Stock Symbol", value="AAPL")
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching data..."):
                df = fetch_stock_data_tiingo(symbol)
                
                if df is not None:
                    st.success(f"Data fetched successfully for {symbol}")
                    st.dataframe(df.head())
                    
                    # Plot the closing price
                    st.subheader("Closing Price")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if 'close' in df.columns:
                        ax.plot(df['close'])
                    elif 'adjClose' in df.columns:
                        ax.plot(df['adjClose'])
                    
                    ax.set_title(f"{symbol} Stock Price")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    st.pyplot(fig)
    else:
        st.header("About")
        st.write(
            "This app uses LSTM neural networks to predict future stock prices. "
            "It is built with Streamlit, TensorFlow, and pandas."
        )

if __name__ == "__main__":
    main()
