"""
LSTM model for Stock Price Prediction
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src import config

def build_model(look_back=None, lstm_units=None, dropout_rate=None):
    """
    Build LSTM model for stock price prediction

    Args:
        look_back (int, optional): Number of previous time steps to use as input features
        lstm_units (int, optional): Number of LSTM units
        dropout_rate (float, optional): Dropout rate

    Returns:
        tensorflow.keras.models.Sequential: LSTM model
    """
    if look_back is None:
        look_back = config.LOOK_BACK

    if lstm_units is None:
        lstm_units = config.LSTM_UNITS

    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def train_model(X_train, Y_train, epochs=None, batch_size=None, validation_split=None, verbose=1):
    """
    Train LSTM model

    Args:
        X_train (numpy.ndarray): Training input data
        Y_train (numpy.ndarray): Training target data
        epochs (int, optional): Number of epochs
        batch_size (int, optional): Batch size
        validation_split (float, optional): Validation split
        verbose (int, optional): Verbosity mode

    Returns:
        tuple: (model, history)
    """
    if epochs is None:
        epochs = config.EPOCHS

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if validation_split is None:
        validation_split = config.VALIDATION_SPLIT

    model = build_model()

    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )

    # Create models directory if it doesn't exist
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    # Save model
    model_path = os.path.join(config.MODEL_PATH, 'lstm_model.h5')
    model.save(model_path)

    return model, history

def load_trained_model(model_path=None):
    """
    Load trained model

    Args:
        model_path (str, optional): Path to model file

    Returns:
        tensorflow.keras.models.Sequential: Trained model
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_PATH, 'lstm_model.h5')

    return load_model(model_path)

def predict(model, X):
    """
    Make predictions with the model

    Args:
        model (tensorflow.keras.models.Sequential): Trained model
        X (numpy.ndarray): Input data

    Returns:
        numpy.ndarray: Predictions
    """
    return model.predict(X)

def inverse_transform(predictions, scaler):
    """
    Inverse transform predictions to original scale

    Args:
        predictions (numpy.ndarray): Predictions
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to normalize data

    Returns:
        numpy.ndarray: Predictions in original scale
    """
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((len(predictions), 1))
    # Put the predictions in the first column
    dummy[:, 0] = predictions.flatten()
    # Inverse transform
    return scaler.inverse_transform(dummy)[:, 0]

def prepare_prediction_data(data, look_back, scaler=None):
    """
    Prepare data for prediction

    Args:
        data (pandas.Series): Stock price data
        look_back (int): Number of previous time steps to use as input features
        scaler (sklearn.preprocessing.MinMaxScaler, optional): Scaler to use. If None, a new one will be created

    Returns:
        tuple: (X, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler

    # Convert to numpy array
    dataset = data.values.reshape(-1, 1)

    # Normalize the dataset
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
    else:
        dataset = scaler.transform(dataset)

    # Use the last 'look_back' data points as input
    X = dataset[-look_back:].reshape(1, look_back, 1)

    return X, scaler

def predict_future(model, last_data, scaler, days=30, look_back=None):
    """
    Predict future stock prices

    Args:
        model (tensorflow.keras.models.Sequential): Trained model
        last_data (numpy.ndarray): Last 'look_back' data points
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to normalize data
        days (int): Number of days to predict
        look_back (int, optional): Number of previous time steps to use as input features

    Returns:
        numpy.ndarray: Predicted prices
    """
    if look_back is None:
        look_back = config.LOOK_BACK

    # Make a copy of the last data
    curr_data = last_data.copy()
    future_predictions = []

    for _ in range(days):
        # Predict the next price
        prediction = predict(model, curr_data)
        future_predictions.append(prediction[0, 0])

        # Update the input data for the next prediction
        curr_data = np.append(curr_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    # Inverse transform to get actual prices
    dummy = np.zeros((len(future_predictions), 1))
    dummy[:, 0] = future_predictions
    future_predictions = scaler.inverse_transform(dummy)[:, 0]

    return future_predictions
