import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import json
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.train_lstm_pipeline import LSTMModel 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_lstm_model(data_dir: Path, model_path: Path, metrics_output_path: Path):
    """
    Loads a trained LSTM model and evaluates it on the test set.
    """
    logging.info("Starting LSTM model evaluation...")
    
    # --- Setup Device (GPU/CPU) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        # --- Load Test Data, Scaler, and Model ---
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy") # This is the true, unscaled data
        scaler = joblib.load(data_dir / "scaler.joblib")
        logging.info("Test data and scaler loaded successfully.")

        # The model needs to know the number of input features
        input_size = X_test.shape[2]
        model = LSTMModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        logging.info("LSTM model loaded successfully.")

        # Convert test data to PyTorch Tensors
        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        # --- Generate Predictions ---
        logging.info("Generating predictions on the test set...")
        with torch.no_grad(): # Disable gradient calculation for inference
            predictions_scaled = model(X_test_tensor).cpu().numpy()

        # --- Inverse Transform Predictions ---
        # The scaler was fitted on multiple features, but we only want to inverse the 'Close' price.
        # We create a dummy array of the same shape as the scaler expects,
        # place our predictions in the 'Close' column (index 3), and then inverse transform.
        num_features = scaler.n_features_in_
        dummy_array = np.zeros((predictions_scaled.shape[0], num_features))
        dummy_array[:, 3] = predictions_scaled.flatten()
        
        predictions_unscaled = scaler.inverse_transform(dummy_array)[:, 3]
        
        # --- Calculate Metrics ---
        mse = mean_squared_error(y_test, predictions_unscaled)
        mae = mean_absolute_error(y_test, predictions_unscaled)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions_unscaled)

        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'r_squared': r2
        }
        
        logging.info("LSTM Model Evaluation Metrics:")
        logging.info(json.dumps(metrics, indent=4))

        # --- Save Metrics ---
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    # Define paths
    SEQUENCE_DATA_DIR = Path("data/sequences")
    MODEL_PATH = Path("models/lstm_v1.pt")
    METRICS_PATH = Path("reports/lstm_metrics.json") # Save to a new file
    
    evaluate_lstm_model(
        data_dir=SEQUENCE_DATA_DIR, 
        model_path=MODEL_PATH, 
        metrics_output_path=METRICS_PATH
    )