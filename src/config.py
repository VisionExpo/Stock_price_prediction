"""
Configuration parameters for the Stock Price Prediction model
"""
import os

# Data parameters
STOCK_SYMBOL = 'AAPL'
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

# Model parameters
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LOOK_BACK = 60  # Number of previous time steps to use as input features

# Training parameters
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
