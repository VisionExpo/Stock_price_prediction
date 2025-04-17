"""
Utility functions for the Stock Price Prediction model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from src import config

def create_dataset(dataset, look_back=1):
    """
    Convert an array of values into a dataset matrix
    
    Args:
        dataset (numpy.ndarray): Array of values
        look_back (int): Number of previous time steps to use as input features
        
    Returns:
        tuple: (X, Y) where X is the input data and Y is the target data
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def preprocess_data(data, look_back, train_size=0.8):
    """
    Preprocess the data for LSTM model
    
    Args:
        data (pandas.DataFrame): Stock price data
        look_back (int): Number of previous time steps to use as input features
        train_size (float): Proportion of data to use for training
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, scaler)
    """
    # Extract the 'close' price and convert to numpy array
    dataset = data.values.reshape(-1, 1)
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # Split into train and test sets
    train_size = int(len(dataset) * train_size)
    train_data = dataset[0:train_size, :]
    test_data = dataset[train_size:, :]
    
    # Create datasets with look_back
    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, Y_train, X_test, Y_test, scaler

def plot_predictions(actual, predicted, title='Stock Price Prediction'):
    """
    Plot actual vs predicted stock prices
    
    Args:
        actual (numpy.ndarray): Actual stock prices
        predicted (numpy.ndarray): Predicted stock prices
        title (str): Title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Create results directory if it doesn't exist
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    
    plt.savefig(os.path.join(config.RESULTS_PATH, f'{title.replace(" ", "_")}.png'))
    plt.show()

def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics
    
    Args:
        actual (numpy.ndarray): Actual stock prices
        predicted (numpy.ndarray): Predicted stock prices
        
    Returns:
        dict: Dictionary of metrics
    """
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def save_results(metrics, filename='metrics.csv'):
    """
    Save metrics to a CSV file
    
    Args:
        metrics (dict): Dictionary of metrics
        filename (str): Name of the file to save
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(config.RESULTS_PATH, filename), index=False)
    print(f"Results saved to {os.path.join(config.RESULTS_PATH, filename)}")
