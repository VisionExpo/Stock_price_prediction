import numpy as np
from tensorflow.keras.models import load_model

def predict_future(model, input_data, n_steps=30):
    """Make future predictions with the model
    
    Args:
        model: Trained LSTM model
        input_data: Input sequence (can be 1D, 2D or 3D array)
        n_steps: Number of future steps to predict
        
    Returns:
        numpy array of predictions
    """
    predictions = []
    expected_sequence_length = model.input_shape[1]
    
    # Convert input to numpy array if it isn't already
    input_data = np.array(input_data)
    
    # Ensure input is in correct shape
    if len(input_data.shape) == 3:
        temp_input = input_data[0, :, 0]
    elif len(input_data.shape) == 2:
        temp_input = input_data[:, 0] if input_data.shape[1] == 1 else input_data[:, -1]
    else:  # 1D array
        temp_input = input_data
    
    # Convert to 1D numpy array
    temp_input = np.array(temp_input).flatten()
    
    # Ensure we have enough data points
    if len(temp_input) < expected_sequence_length:
        raise ValueError(f"Input sequence length {len(temp_input)} is shorter than required {expected_sequence_length}")
    
    # Make predictions
    for _ in range(n_steps):
        # Take the last expected_sequence_length elements
        sequence = temp_input[-expected_sequence_length:]
        x_input = sequence.reshape((1, expected_sequence_length, 1))
        
        # Make prediction
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0, 0])
        temp_input = np.append(temp_input, yhat[0, 0])
    
    return np.array(predictions)

def save_model(model, filename):
    model.save(filename)

if __name__ == "__main__":
    # Example usage
    pass
