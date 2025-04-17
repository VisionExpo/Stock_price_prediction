"""
Integration tests for the complete pipeline
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from src.data_loader import prepare_data
from src.model import train_model, predict, inverse_transform
from src.utils import calculate_metrics
from src import config

class TestPipeline(unittest.TestCase):
    """
    Test cases for the complete pipeline
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create temporary directories
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_model_path = config.MODEL_PATH
        self.original_results_path = config.RESULTS_PATH
        config.MODEL_PATH = os.path.join(self.temp_dir.name, 'models')
        config.RESULTS_PATH = os.path.join(self.temp_dir.name, 'results')
        
        # Create directories
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        
        # Create sample data (sine wave with noise)
        x = np.linspace(0, 10, 200)
        y = np.sin(x) + 0.1 * np.random.randn(len(x))
        self.sample_data = pd.Series(y)
    
    def tearDown(self):
        """
        Clean up after tests
        """
        # Restore original paths
        config.MODEL_PATH = self.original_model_path
        config.RESULTS_PATH = self.original_results_path
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_full_pipeline(self):
        """
        Test the complete pipeline from data preparation to prediction
        """
        # 1. Prepare data
        look_back = 10
        X_train, Y_train, X_test, Y_test, scaler = prepare_data(
            df=pd.DataFrame({'close': self.sample_data}),
            look_back=look_back,
            train_size=0.8
        )
        
        # 2. Train model
        model, history = train_model(
            X_train, Y_train,
            epochs=5,  # Small number for testing
            batch_size=8,
            validation_split=0.1,
            verbose=0
        )
        
        # 3. Make predictions
        predictions = predict(model, X_test)
        
        # 4. Inverse transform
        actual_prices = inverse_transform(Y_test, scaler)
        predicted_prices = inverse_transform(predictions, scaler)
        
        # 5. Calculate metrics
        metrics = calculate_metrics(actual_prices, predicted_prices)
        
        # Check if model was saved
        self.assertTrue(os.path.exists(os.path.join(config.MODEL_PATH, 'lstm_model.h5')))
        
        # Check metrics
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('MAPE', metrics)
        
        # Check predictions
        self.assertEqual(len(predicted_prices), len(actual_prices))

if __name__ == '__main__':
    unittest.main()
