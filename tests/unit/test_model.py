"""
Unit tests for the model module
"""

import unittest
import os
import numpy as np
import tempfile
from sklearn.preprocessing import MinMaxScaler
from src.model import build_model, predict, inverse_transform, prepare_prediction_data

class TestModel(unittest.TestCase):
    """
    Test cases for model functions
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create sample data
        self.look_back = 5
        self.X_train = np.random.rand(10, self.look_back, 1)
        self.Y_train = np.random.rand(10, 1)
        
        # Create a scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(np.random.rand(100, 1))
        
    def test_build_model(self):
        """
        Test build_model function
        """
        # Build model
        model = build_model(look_back=self.look_back)
        
        # Check model structure
        self.assertEqual(model.input_shape, (None, self.look_back, 1))
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_predict(self):
        """
        Test predict function
        """
        # Build model
        model = build_model(look_back=self.look_back)
        
        # Make predictions
        predictions = predict(model, self.X_train)
        
        # Check predictions shape
        self.assertEqual(predictions.shape, (len(self.X_train), 1))
    
    def test_inverse_transform(self):
        """
        Test inverse_transform function
        """
        # Create sample predictions
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(-1, 1)
        
        # Inverse transform
        original_predictions = inverse_transform(predictions, self.scaler)
        
        # Check shape
        self.assertEqual(original_predictions.shape, (len(predictions),))
        
        # Check values
        dummy = np.zeros((len(predictions), 1))
        dummy[:, 0] = predictions.flatten()
        expected = self.scaler.inverse_transform(dummy)[:, 0]
        np.testing.assert_array_equal(original_predictions, expected)
    
    def test_prepare_prediction_data(self):
        """
        Test prepare_prediction_data function
        """
        # Create sample data
        data = np.sin(np.linspace(0, 10, 100))
        data_series = pd.Series(data)
        
        # Prepare data
        X, scaler = prepare_prediction_data(data_series, self.look_back)
        
        # Check shapes
        self.assertEqual(X.shape, (1, self.look_back, 1))
        
        # Check scaler
        self.assertIsInstance(scaler, MinMaxScaler)

if __name__ == '__main__':
    unittest.main()
