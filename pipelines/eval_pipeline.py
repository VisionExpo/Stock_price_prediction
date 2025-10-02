import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# UPDATED: Import TransformerModel instead of LSTMModel
from pipelines.train_transformer_pipeline import TransformerModel 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_dir: Path, model_path: Path, metrics_output_path: Path):
    logging.info("Starting Transformer model evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load test data and scalers
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy") 
        y_scaler = joblib.load(data_dir / "y_scaler.joblib")
        x_scaler = joblib.load(data_dir / "x_scaler.joblib")
        logging.info("Test data and scalers loaded successfully.")

        # UPDATED: Instantiate TransformerModel
        input_size = X_test.shape[2]
        model = TransformerModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info("Transformer model loaded successfully.")

        X_test_tensor = torch.from_numpy(X_test).float().to(device)

        # Generate predictions
        with torch.no_grad():
            predictions_scaled = model(X_test_tensor).cpu().numpy()

        # Inverse transform predictions
        predictions_unscaled = y_scaler.inverse_transform(predictions_scaled)
        
        # Calculate Metrics
        metrics = {
            'mean_squared_error': mean_squared_error(y_test, predictions_unscaled),
            'mean_absolute_error': mean_absolute_error(y_test, predictions_unscaled),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, predictions_unscaled)),
            'r_squared': r2_score(y_test, predictions_unscaled)
        }
        
        logging.info("Transformer Model Evaluation Metrics:")
        logging.info(json.dumps(metrics, indent=4))

        # Save Metrics
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    # UPDATED: Point to the new data, model, and metrics files
    SEQUENCE_DATA_DIR = Path("data/sequences_sentiment")
    MODEL_PATH = Path("models/transformer_v1.pt")
    METRICS_PATH = Path("reports/transformer_metrics.json")
    
    evaluate_model(
        data_dir=SEQUENCE_DATA_DIR, 
        model_path=MODEL_PATH, 
        metrics_output_path=METRICS_PATH
    )