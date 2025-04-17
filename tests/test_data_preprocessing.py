import unittest
import numpy as np
from src.data_preprocessing import preprocess_data, create_dataset

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        # Test if preprocessing works correctly
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        train_data, test_data, scaler = preprocess_data(data)
        self.assertEqual(len(train_data), 7, "Training data length should be 70% of input data.")
        self.assertEqual(len(test_data), 3, "Testing data length should be 30% of input data.")

    def test_create_dataset(self):
        # Test if dataset creation works correctly
        data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        x, y = create_dataset(data, time_step=3)
        self.assertEqual(x.shape[0], 7, "X should have 7 samples.")
        self.assertEqual(y.shape[0], 7, "Y should have 7 samples.")

if __name__ == "__main__":
    unittest.main()
