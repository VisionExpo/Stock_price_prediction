import unittest
import requests
import os
import json
import pandas as pd
from src.utils import check_system_health, get_cached_stock_data
from src.data_retrieval import retrieve_data

class TestDeployment(unittest.TestCase):
    def setUp(self):
        """Start the Streamlit app for testing"""
        # The app should be running separately
        self.base_url = "http://localhost:8501"
        
    def test_health_check(self):
        """Test health check endpoint"""
        health_metrics = check_system_health()
        self.assertEqual(health_metrics['status'], 'healthy')
        self.assertIn('memory_usage', health_metrics)
        self.assertIn('cpu_usage', health_metrics)
        
    def test_data_caching(self):
        """Test data caching functionality"""
        # First call should create cache
        data1 = get_cached_stock_data('AAPL', 100)
        self.assertIsInstance(data1, pd.DataFrame)
        self.assertFalse(data1.empty)
        
        # Second call should use cache
        data2 = get_cached_stock_data('AAPL', 100)
        pd.testing.assert_frame_equal(data1, data2)
        
    def test_model_file(self):
        """Test if model file exists and is valid"""
        self.assertTrue(os.path.exists('stock_model.h5'))
        self.assertGreater(os.path.getsize('stock_model.h5'), 0)
        
    def test_metrics_storage(self):
        """Test metrics storage functionality"""
        self.assertTrue(os.path.exists('model_metrics'))
        metrics_files = os.listdir('model_metrics')
        if metrics_files:  # If there are any metric files
            with open(os.path.join('model_metrics', metrics_files[0])) as f:
                metrics = json.load(f)
            self.assertIn('metrics', metrics)
            self.assertIn('parameters', metrics)
            
    def test_cache_directory(self):
        """Test cache directory setup"""
        self.assertTrue(os.path.exists('cache'))
        
if __name__ == '__main__':
    unittest.main()