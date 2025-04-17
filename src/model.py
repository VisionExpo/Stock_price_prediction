import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, lstm_units=100, dropout_rate=0.2, learning_rate=0.001):
    """
    Create LSTM model with configurable hyperparameters
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        lstm_units (int): Number of LSTM units in each layer
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
    """
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64):
    """Train the model with the given parameters"""
    return model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=epochs, batch_size=batch_size, verbose=1)

def predict_future(model, input_data, n_steps=30):
    """Make future predictions"""
    predictions = []
    temp_input = list(input_data)
    
    # Get the length of input sequence the model expects
    expected_sequence_length = model.input_shape[1]
    
    for _ in range(n_steps):
        x_input = np.array(temp_input[-expected_sequence_length:])
        x_input = x_input.reshape((1, expected_sequence_length, 1))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0, 0])
        temp_input.append(yhat[0, 0])
    
    return np.array(predictions)

if __name__ == "__main__":
    # Example usage
    pass
