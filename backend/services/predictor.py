import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.train_transformer_pipeline import TransformerModel

# This dictionary will act as a global cache for loaded resources
cache = {}

def load_resources():
    """Loads all necessary ML resources into the cache."""
    if "model" in cache:
        logging.info("Resources already loaded.")
        return

    logging.info("Loading resources for Transformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache["device"] = device
    
    try:
        cache["x_scaler"] = joblib.load(Path("data/sequences_sentiment/x_scaler.joblib"))
        cache["y_scaler"] = joblib.load(Path("data/sequences_sentiment/y_scaler.joblib"))
        data_path = Path("data/processed/final_fused_data.csv")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        cache["data"] = df
        
        model_path = Path("models/transformer_v1.pt")
        input_size = cache["x_scaler"].n_features_in_
        
        model = TransformerModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        cache["model"] = model
        logging.info("Transformer model and all resources loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load resources on startup: {e}")

def make_prediction(ticker: str, date_str: str) -> float:
    """Makes a single stock prediction for a given ticker and date."""
    if "model" not in cache:
        raise ValueError("Model and resources are not loaded.")

    model = cache["model"]
    data_df = cache["data"]
    x_scaler = cache["x_scaler"]
    y_scaler = cache["y_scaler"]
    device = cache["device"]

    sequence_length = 60
    end_date = pd.to_datetime(date_str)
    
    ticker_data = data_df[data_df['Ticker'] == ticker.upper()]
    if ticker_data.empty:
        raise FileNotFoundError(f"Data for ticker '{ticker}' not found.")

    data_up_to_date = ticker_data[ticker_data['Date'] <= end_date]
    if len(data_up_to_date) < sequence_length:
        raise ValueError(f"Not enough historical data for {ticker} before {date_str}.")

    sequence_to_predict = data_up_to_date.tail(sequence_length)
    feature_cols = x_scaler.feature_names_in_
    sequence_scaled = x_scaler.transform(sequence_to_predict[feature_cols])
    
    input_tensor = torch.from_numpy(sequence_scaled).float().unsqueeze(0).to(device)
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    prediction_unscaled = y_scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
    return prediction_unscaled