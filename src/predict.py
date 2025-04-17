import numpy as np
from tensorflow.keras.models import load_model

def predict_future(model, input_data, n_steps=30):
    predictions = []
    temp_input = list(input_data)
    
    # Get the length of input sequence the model expects from the input shape
    expected_sequence_length = model.input_shape[1]
    
    for i in range(n_steps):
        x_input = np.array(temp_input[-expected_sequence_length:])
        x_input = x_input.reshape((1, expected_sequence_length, 1))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0, 0])
        temp_input.append(yhat[0, 0])
    
    return np.array(predictions)

def save_model(model, filename):
    model.save(filename)

if __name__ == "__main__":
    # Example usage
    pass
