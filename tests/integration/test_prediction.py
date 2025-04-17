"""
Integration tests for the prediction functionality
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from src.model import build_model, train_model, predict_future, prepare_prediction_data
from src import config

class TestPrediction(unittest.TestCase):
    """
    Test cases for the prediction functionality
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_model_path = config.MODEL_PATH
        config.MODEL_PATH = os.path.join(self.temp_dir.name, 'models')
        
        # Create directories
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        
        # Create sample data (sine wave)
        x = np.linspace(0, 10, 200)
        y = np.sin(x)
        self.sample_data = pd.Series(y)
        
        # Parameters
        self.look_back = 10
        
        # Prepare data for training
        dataset = self.sample_data.values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(dataset)
        
        # Create training data
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        X, Y = np.array(X), np.array(Y)
        
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Train a simple model
        self.model = build_model(look_back=self.look_back, lstm_units=10)
        self.model.fit(X, Y, epochs=5, batch_size=8, verbose=0)
    
    def tearDown(self):
        """
        Clean up after tests
        """
        # Restore original paths
        config.MODEL_PATH = self.original_model_path
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_future_prediction(self):
        """
        Test future prediction functionality
        """
        # Prepare data for prediction
        X, scaler = prepare_prediction_data(self.sample_data, self.look_back, self.scaler)
        
        # Predict future values
        future_days = 30
        future_predictions = predict_future(self.model, X, scaler, future_days, self.look_back)
        
        # Check predictions
        self.assertEqual(len(future_predictions), future_days)
        
        # Check if predictions are within a reasonable range
        # For sine wave, values should be between -1 and 1
        self.assertTrue(np.all(future_predictions >= -1.5))
        self.assertTrue(np.all(future_predictions <= 1.5))

if __name__ == '__main__':
    unittest.main()
