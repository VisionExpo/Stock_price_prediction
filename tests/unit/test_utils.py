"""
Unit tests for the utils module
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils import create_dataset, preprocess_data, calculate_metrics

class TestUtils(unittest.TestCase):
    """
    Test cases for utility functions
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create sample data
        self.data = pd.Series(np.sin(np.linspace(0, 10, 100)) + 5)  # Sine wave with offset
        self.look_back = 5
        
    def test_create_dataset(self):
        """
        Test create_dataset function
        """
        # Convert data to numpy array
        dataset = self.data.values.reshape(-1, 1)
        
        # Create dataset
        X, Y = create_dataset(dataset, self.look_back)
        
        # Check shapes
        self.assertEqual(X.shape[0], len(dataset) - self.look_back)
        self.assertEqual(X.shape[1], self.look_back)
        self.assertEqual(Y.shape[0], len(dataset) - self.look_back)
        
        # Check values
        for i in range(len(X)):
            self.assertTrue(np.array_equal(X[i], dataset[i:i+self.look_back, 0]))
            self.assertEqual(Y[i], dataset[i+self.look_back, 0])
    
    def test_preprocess_data(self):
        """
        Test preprocess_data function
        """
        # Preprocess data
        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(self.data, self.look_back, train_size=0.8)
        
        # Check shapes
        train_size = int(len(self.data) * 0.8)
        expected_train_samples = train_size - self.look_back
        expected_test_samples = len(self.data) - train_size - self.look_back
        
        self.assertEqual(X_train.shape[0], expected_train_samples)
        self.assertEqual(X_train.shape[1], self.look_back)
        self.assertEqual(X_train.shape[2], 1)  # Feature dimension
        self.assertEqual(Y_train.shape[0], expected_train_samples)
        
        self.assertEqual(X_test.shape[0], expected_test_samples)
        self.assertEqual(X_test.shape[1], self.look_back)
        self.assertEqual(X_test.shape[2], 1)  # Feature dimension
        self.assertEqual(Y_test.shape[0], expected_test_samples)
        
        # Check scaler
        self.assertIsInstance(scaler, MinMaxScaler)
        
    def test_calculate_metrics(self):
        """
        Test calculate_metrics function
        """
        # Create sample actual and predicted values
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        
        # Calculate metrics
        metrics = calculate_metrics(actual, predicted)
        
        # Check metrics
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE', metrics)
        
        # Check values
        self.assertAlmostEqual(metrics['MSE'], np.mean((actual - predicted) ** 2), places=6)
        self.assertAlmostEqual(metrics['RMSE'], np.sqrt(np.mean((actual - predicted) ** 2)), places=6)
        self.assertAlmostEqual(metrics['MAE'], np.mean(np.abs(actual - predicted)), places=6)
        self.assertAlmostEqual(metrics['MAPE'], np.mean(np.abs((actual - predicted) / actual)) * 100, places=6)

if __name__ == '__main__':
    unittest.main()
