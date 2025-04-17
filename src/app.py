"""
Streamlit web application for Stock Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from dotenv import load_dotenv
import yfinance as yf

from src.data_loader import load_stock_data, fetch_stock_data
from src.model import load_trained_model, predict_future, prepare_prediction_data
from src import config

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Create directories if they don't exist
for directory in [config.DATA_PATH, config.MODEL_PATH, config.RESULTS_PATH]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_stock_data_yf(symbol, period="2y"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
        period (str): Period to fetch data for
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        # Save data to CSV
        if not os.path.exists(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
        
        df.to_csv(os.path.join(config.DATA_PATH, f'{symbol}.csv'))
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def load_or_fetch_data(symbol):
    """
    Load stock data from CSV or fetch from API if file doesn't exist
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        pandas.DataFrame: Stock data
    """
    filepath = os.path.join(config.DATA_PATH, f'{symbol}.csv')
    
    # Check if file exists and is not older than 1 day
    if os.path.exists(filepath):
        file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_age.days < 1:
            df = pd.read_csv(filepath)
            return df
    
    # File doesn't exist or is too old, fetch from API
    return fetch_stock_data_yf(symbol)

def train_model_ui():
    """
    UI for training the model
    """
    st.header("Train Model")
    
    with st.form("train_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Stock Symbol", value=config.STOCK_SYMBOL)
            look_back = st.slider("Look Back Period", min_value=10, max_value=100, value=config.LOOK_BACK)
            epochs = st.slider("Epochs", min_value=10, max_value=200, value=config.EPOCHS)
        
        with col2:
            batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=config.BATCH_SIZE)
            train_test_split = st.slider("Train-Test Split", min_value=0.5, max_value=0.9, value=config.TRAIN_TEST_SPLIT)
            validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=config.VALIDATION_SPLIT)
        
        submit_button = st.form_submit_button("Train Model")
    
    if submit_button:
        st.info("Training model... This may take a few minutes.")
        
        # Load data
        df = load_or_fetch_data(symbol)
        if df is None:
            st.error("Failed to load data. Please try again.")
            return
        
        # Preprocess data
        from src.data_loader import prepare_data
        X_train, Y_train, X_test, Y_test, scaler = prepare_data(
            df, look_back=look_back, train_size=train_test_split
        )
        
        # Train model
        from src.model import train_model
        model, history = train_model(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Plot training history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        
        st.pyplot(fig)
        
        # Evaluate model
        from src.model import predict, inverse_transform
        from src.utils import calculate_metrics
        
        predictions = predict(model, X_test)
        actual_prices = inverse_transform(Y_test, scaler)
        predicted_prices = inverse_transform(predictions, scaler)
        
        metrics = calculate_metrics(actual_prices, predicted_prices)
        
        st.subheader("Model Performance")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_prices, label='Actual')
        ax.plot(predicted_prices, label='Predicted')
        ax.set_title(f"{symbol} Stock Price Prediction")
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        
        st.pyplot(fig)
        
        st.success(f"Model trained and saved to {os.path.join(config.MODEL_PATH, 'lstm_model.h5')}")

def predict_ui():
    """
    UI for making predictions
    """
    st.header("Predict Future Stock Prices")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Stock Symbol", value=config.STOCK_SYMBOL)
            look_back = st.slider("Look Back Period", min_value=10, max_value=100, value=config.LOOK_BACK)
        
        with col2:
            future_days = st.slider("Days to Predict", min_value=5, max_value=60, value=30)
            model_path = st.text_input("Model Path", value=os.path.join(config.MODEL_PATH, "lstm_model.h5"))
        
        submit_button = st.form_submit_button("Make Prediction")
    
    if submit_button:
        # Check if model exists
        if not os.path.exists(model_path):
            st.error("Model not found. Please train a model first.")
            return
        
        # Load data
        df = load_or_fetch_data(symbol)
        if df is None:
            st.error("Failed to load data. Please try again.")
            return
        
        # Load model
        model = load_trained_model(model_path)
        
        # Extract the 'close' price
        if 'Close' in df.columns:
            data = df['Close']
        elif 'close' in df.columns:
            data = df['close']
        else:
            # If the dataframe has a multi-level index, reset it
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            if 'Close' in df.columns:
                data = df['Close']
            else:
                data = df['close']
        
        # Prepare data for prediction
        X, scaler = prepare_prediction_data(data, look_back)
        
        # Predict future prices
        future_predictions = predict_future(model, X, scaler, future_days, look_back)
        
        # Create dates for future predictions
        if 'Date' in df.columns:
            last_date = pd.to_datetime(df['Date'].iloc[-1])
        else:
            last_date = pd.to_datetime(df.index[-1])
        
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(future_days)]
        
        # Create a dataframe with the predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions
        })
        
        st.subheader("Future Price Predictions")
        st.dataframe(future_df)
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(data.values[-100:], label='Historical')
        
        # Plot future predictions
        ax.plot(range(len(data)-1, len(data)+future_days-1), future_predictions, label='Predicted')
        
        ax.set_title(f"{symbol} Stock Price Prediction")
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        
        st.pyplot(fig)
        
        # Save predictions to CSV
        predictions_file = os.path.join(config.RESULTS_PATH, f'{symbol}_future_predictions.csv')
        future_df.to_csv(predictions_file, index=False)
        st.success(f"Predictions saved to {predictions_file}")

def main():
    """
    Main function for the Streamlit app
    """
    st.title("Stock Price Prediction App")
    st.write("Predict future stock prices using LSTM neural networks")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Predict"])
    
    if page == "Train Model":
        train_model_ui()
    else:
        predict_ui()
    
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses LSTM neural networks to predict future stock prices. "
        "It is built with Streamlit, TensorFlow, and pandas."
    )

if __name__ == "__main__":
    main()
