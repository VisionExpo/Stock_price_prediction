import unittest
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from src.predict import predict_future

class TestPredict(unittest.TestCase):
    def test_predict_future(self):
        # Test if future predictions can be made without errors
        model = Sequential()  # Placeholder for the model
        model.add(Dense(1, input_shape=(100, 1)))  # Mock model structure
        input_data = np.random.rand(100).reshape(1, 100, 1)  # Simulated input data
        predictions = predict_future(model, input_data, n_steps=5)
        self.assertEqual(len(predictions), 5, "Predictions should match the number of steps requested.")

if __name__ == "__main__":
    unittest.main()
