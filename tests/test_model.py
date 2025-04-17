import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from src.model import create_model, train_model

class TestModel(unittest.TestCase):
    def test_create_model(self):
        # Test if the model is created correctly
        model = create_model((100, 1))
        self.assertIsInstance(model, Sequential, "Model should be an instance of Sequential.")

    def test_train_model(self):
        # Test if the model can be trained without errors
        x_train = np.random.rand(100, 100, 1)
        y_train = np.random.rand(100, 1)
        x_test = np.random.rand(20, 100, 1)
        y_test = np.random.rand(20, 1)
        
        model = create_model((100, 1))
        trained_model = train_model(model, x_train, y_train, x_test, y_test)
        self.assertIsNotNone(trained_model, "Trained model should not be None.")

if __name__ == "__main__":
    unittest.main()
