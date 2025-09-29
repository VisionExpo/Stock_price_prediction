import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.train_lstm_pipeline import LSTMModel 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: Loading resources...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache["device"] = device
    
    try:
        cache["x_scaler"] = joblib.load(Path("data/sequences/x_scaler.joblib"))
        cache["y_scaler"] = joblib.load(Path("data/sequences/y_scaler.joblib"))
        data_path = Path("data/processed/final_fused_data.csv")
        # --- UPDATED: Load data as a flat DataFrame without setting the index ---
        cache["data"] = pd.read_csv(data_path, parse_dates=['Date'])
        model_path = Path("models/lstm_v2_sentiment.pt")
        input_size = cache["x_scaler"].n_features_in_
        model = LSTMModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        cache["model"] = model
        logging.info("All resources loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load resources on startup: {e}")

    yield
    logging.info("Application shutdown: Clearing cache...")
    cache.clear()

app = FastAPI(title="Stock Market AI API (LSTM)", lifespan=lifespan)

class PredictionRequest(BaseModel):
    ticker: str
    date: str

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market AI API!"}

@app.get("/history/{ticker}")
def get_history(ticker: str):
    data_df = cache.get("data")
    if data_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    
    # --- UPDATED: Use a robust boolean filter ---
    ticker_data = data_df[data_df['Ticker'] == ticker.upper()]
    if ticker_data.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found in dataset.")
    
    history = ticker_data[['Date', 'Close']]
    return history.to_dict('records')

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # ... (retrieve model, scalers, etc. from cache) ...
    model = cache.get("model"); data_df = cache.get("data"); x_scaler = cache.get("x_scaler"); y_scaler = cache.get("y_scaler"); device = cache.get("device")
    if not all([model, data_df is not None, x_scaler, y_scaler]):
        raise HTTPException(status_code=500, detail="Server resources not loaded.")

    try:
        sequence_length = 60
        end_date = pd.to_datetime(request.date)
        ticker = request.ticker.upper()
        
        # --- UPDATED: Use a robust boolean filter ---
        ticker_data = data_df[data_df['Ticker'] == ticker]
        if ticker_data.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found in dataset.")

        data_up_to_date = ticker_data[ticker_data['Date'] <= end_date].set_index('Date')

        if len(data_up_to_date) < sequence_length:
            raise HTTPException(status_code=400, detail=f"Not enough historical data to make a prediction.")
        
        sequence_to_predict = data_up_to_date.tail(sequence_length)
        feature_cols = x_scaler.feature_names_in_
        sequence_scaled = x_scaler.transform(sequence_to_predict[feature_cols])
        
        input_tensor = torch.from_numpy(sequence_scaled).float().unsqueeze(0).to(device)
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
        
        prediction_unscaled = y_scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
        
        return {"predicted_price": prediction_unscaled}
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))