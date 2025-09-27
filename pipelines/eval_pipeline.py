import torch
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add project root to path to find pipelines package
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.train_lstm_pipeline import LSTMModel 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_lstm_model(data_dir: Path, model_path: Path, metrics_output_path: Path):
    logging.info("Starting LSTM model evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # --- UPDATED: Load Test Data and both scalers ---
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy") # This is the true, unscaled data
        y_scaler = joblib.load(data_dir / "y_scaler.joblib")
        logging.info("Test data and scalers loaded successfully.")

        input_size = X_test.shape[2]
        model = LSTMModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        # --- Generate Predictions ---
        with torch.no_grad():
            predictions_scaled = model(X_test_tensor).cpu().numpy()

        # --- UPDATED: Inverse Transform Predictions using y_scaler ---
        predictions_unscaled = y_scaler.inverse_transform(predictions_scaled)
        
        # --- Calculate Metrics ---
        metrics = {
            'mean_squared_error': mean_squared_error(y_test, predictions_unscaled),
            'mean_absolute_error': mean_absolute_error(y_test, predictions_unscaled),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, predictions_unscaled)),
            'r_squared': r2_score(y_test, predictions_unscaled)
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
    SEQUENCE_DATA_DIR = Path("data/sequences")
    MODEL_PATH = Path("models/lstm_v1.pt")
    METRICS_PATH = Path("reports/lstm_metrics.json")
    
    evaluate_lstm_model(
        data_dir=SEQUENCE_DATA_DIR, 
        model_path=MODEL_PATH, 
        metrics_output_path=METRICS_PATH
    )