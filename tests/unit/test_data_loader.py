"""
Unit tests for the data_loader module
"""

import unittest
import os
import pandas as pd
import numpy as np
import tempfile
from src.data_loader import load_stock_data, prepare_data
from src import config

class TestDataLoader(unittest.TestCase):
    """
    Test cases for data loader functions
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_path = config.DATA_PATH
        config.DATA_PATH = self.temp_dir.name
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Save sample data
        os.makedirs(config.DATA_PATH, exist_ok=True)
        self.sample_data.to_csv(os.path.join(config.DATA_PATH, 'AAPL.csv'), index=False)
    
    def tearDown(self):
        """
        Clean up after tests
        """
        # Restore original data path
        config.DATA_PATH = self.original_data_path
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_load_stock_data(self):
        """
        Test load_stock_data function
        """
        # Load data
        df = load_stock_data('AAPL')
        
        # Check if data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn('close', df.columns)
    
    def test_prepare_data(self):
        """
        Test prepare_data function
        """
        # Prepare data
        look_back = 10
        X_train, Y_train, X_test, Y_test, scaler = prepare_data(
            df=self.sample_data,
            look_back=look_back,
            train_size=0.8
        )
        
        # Check shapes
        train_size = int(len(self.sample_data) * 0.8)
        expected_train_samples = train_size - look_back
        expected_test_samples = len(self.sample_data) - train_size - look_back
        
        self.assertEqual(X_train.shape[0], expected_train_samples)
        self.assertEqual(X_train.shape[1], look_back)
        self.assertEqual(X_train.shape[2], 1)  # Feature dimension
        self.assertEqual(Y_train.shape[0], expected_train_samples)
        
        self.assertEqual(X_test.shape[0], expected_test_samples)
        self.assertEqual(X_test.shape[1], look_back)
        self.assertEqual(X_test.shape[2], 1)  # Feature dimension
        self.assertEqual(Y_test.shape[0], expected_test_samples)

if __name__ == '__main__':
    unittest.main()
