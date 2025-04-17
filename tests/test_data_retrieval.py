import unittest
from src.data_retrieval import retrieve_data

class TestDataRetrieval(unittest.TestCase):
    def test_retrieve_data(self):
        # Test if data retrieval works for a known ticker
        df = retrieve_data('AAPL')
        self.assertFalse(df.empty, "DataFrame should not be empty for valid ticker.")

if __name__ == "__main__":
    unittest.main()
