import pandas as pd
from pathlib import Path
import logging
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_path: Path, model_path: Path, metrics_output_path: Path):
    """
    Loads a trained model and evaluates it on a test set split chronologically.

    Args:
        data_path (Path): Path to the processed data CSV file.
        model_path (Path): Path to the trained model file.
        metrics_output_path (Path): Path to save the evaluation metrics JSON file.
    """
    logging.info("Starting model evaluation...")

    try:
        # Load the processed data and the trained model
        df = pd.read_csv(data_path, parse_dates=['Date'])
        model = joblib.load(model_path)
        logging.info(f"Data and model loaded successfully.")

        # ----- Chronological Train-Test Split -----
        # A more realistic split for time series data
        test_start_date = "2023-01-01"
        test_df = df[df['Date'] >= test_start_date].copy()
        

        if test_df.empty:
            logging.error("No data available for testing in the specified date range. Aborting.")
            return
        
        logging.info(f"Test set contains {len(test_df)} rows, from {test_df['Date'].min()} to {test_df['Date'].max()}.")

        # Create the target variable for the test set
        test_df['target'] = test_df.groupby('Ticker')['Close'].shift(-1)
        test_df.dropna(subset=['target'], inplace=True)

        # Define features (X) and target (y)
        features_to_exclude = ['Date', 'Ticker', 'target']
        features = [col for col in test_df.columns if col not in features_to_exclude]

        X_test = test_df[features]
        y_test = test_df['target']

        # ----- Mke Predictions and Calculate Metrics -----
        logging.info("Making predictions on the test set...")
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'r_squared': r2
        }

        logging.info("Evaluation Metrics:")
        logging.info(json.dumps(metrics, indent=4))

        # ----- Save Metrics -----
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_output_path}.")

    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")


if __name__ == "__main__":
    # Define paths
    PROCESSED_DATA_PATH = Path("data/processed/processed_market_data.csv")
    MODEL_PATH = Path("models/random_forest_honest_baseline.joblib")
    METRICS_PATH = Path("reports/metrics.json")

    # Execute the evaluation function
    evaluate_model(
        data_path=PROCESSED_DATA_PATH,
        model_path=MODEL_PATH,
        metrics_output_path=METRICS_PATH
    )