"""
Data loading and preprocessing for the Stock Price Prediction model
"""

import os
import pandas as pd
import pandas_datareader as pdr
from dotenv import load_dotenv
import numpy as np
from src.utils import preprocess_data
from src import config

def load_api_key():
    """
    Load API key from environment variables
    
    Returns:
        str: API key
    """
    load_dotenv()
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError("TIINGO_API_KEY not found in environment variables")
    return api_key

def fetch_stock_data(symbol, api_key=None):
    """
    Fetch stock data from Tiingo API
    
    Args:
        symbol (str): Stock symbol
        api_key (str, optional): Tiingo API key. If None, will try to load from environment
        
    Returns:
        pandas.DataFrame: Stock data
    """
    if api_key is None:
        api_key = load_api_key()
    
    try:
        df = pdr.get_data_tiingo(symbol, api_key=api_key)
        # Create data directory if it doesn't exist
        if not os.path.exists(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
        # Save data to CSV
        df.to_csv(os.path.join(config.DATA_PATH, f'{symbol}.csv'))
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def load_stock_data(symbol=None, filepath=None):
    """
    Load stock data from CSV file or fetch from API if file doesn't exist
    
    Args:
        symbol (str, optional): Stock symbol. Required if filepath is None
        filepath (str, optional): Path to CSV file. If None, will use default path
        
    Returns:
        pandas.DataFrame: Stock data
    """
    if filepath is None:
        if symbol is None:
            symbol = config.STOCK_SYMBOL
        filepath = os.path.join(config.DATA_PATH, f'{symbol}.csv')
    
    # Check if file exists
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df
    else:
        # File doesn't exist, fetch from API
        if symbol is None:
            raise ValueError("Symbol must be provided if file doesn't exist")
        return fetch_stock_data(symbol)

def prepare_data(df=None, symbol=None, look_back=None, train_size=None):
    """
    Prepare data for LSTM model
    
    Args:
        df (pandas.DataFrame, optional): Stock data. If None, will load from file
        symbol (str, optional): Stock symbol. Required if df is None
        look_back (int, optional): Number of previous time steps to use as input features
        train_size (float, optional): Proportion of data to use for training
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, scaler)
    """
    if df is None:
        df = load_stock_data(symbol)
    
    if look_back is None:
        look_back = config.LOOK_BACK
    
    if train_size is None:
        train_size = config.TRAIN_TEST_SPLIT
    
    # Extract the 'close' price
    if 'close' in df.columns:
        data = df['close']
    else:
        # If the dataframe has a multi-level index, reset it
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        data = df['close']
    
    return preprocess_data(data, look_back, train_size)
