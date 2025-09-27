import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys

# --- Add project root to path to find pipelines package ---
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the LSTMModel class from the training script
from pipelines.train_lstm_pipeline import LSTMModel 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- App Cache ---
# Will hold our model, data, and scalers
cache = {}

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: Loading LSTM model, data, and scalers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache["device"] = device
    
    # Load scalers
    try:
        cache["x_scaler"] = joblib.load(Path("data/sequences/x_scaler.joblib"))
        cache["y_scaler"] = joblib.load(Path("data/sequences/y_scaler.joblib"))
        logging.info("Scalers loaded successfully.")
    except FileNotFoundError:
        cache["x_scaler"] = None
        cache["y_scaler"] = None
        logging.error("Scaler files not found.")

    # Load data
    data_path = Path("data/processed/processed_market_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=['Date'])
        # Set a multi-index for faster lookups
        cache["data"] = df.set_index(['Ticker', 'Date'])
        logging.info("Data loaded and indexed successfully.")
    else:
        cache["data"] = None
        logging.error(f"Data file not found at {data_path}")

    # Load LSTM model
    model_path = Path("models/lstm_v2_sentiment.pt")
    if model_path.exists():
        # Determine input size from the scaler
        input_size = cache["x_scaler"].n_features_in_
        model = LSTMModel(input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        cache["model"] = model
        logging.info("LSTM model loaded successfully.")
    else:
        cache["model"] = None
        logging.error(f"Model file not found at {model_path}")
        
    yield
    
    logging.info("Application shutdown: Clearing cache...")
    cache.clear()

app = FastAPI(
    title="Stock Market AI API (LSTM)",
    description="API for predicting stock prices using an LSTM model.",
    lifespan=lifespan
)

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    ticker: str
    date: str # Format: 'YYYY-MM-DD'

class PredictionResponse(BaseModel):
    predicted_price: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market AI API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Retrieve everything from the cache
    model = cache.get("model")
    data_df = cache.get("data")
    x_scaler = cache.get("x_scaler")
    y_scaler = cache.get("y_scaler")
    device = cache.get("device")

    if not all([model, data_df is not None, x_scaler, y_scaler]):
        raise HTTPException(status_code=500, detail="Server resources not loaded. Check logs.")

    try:
        # --- Prepare the input sequence ---
        sequence_length = 60 # This must match the sequence length from training
        end_date = pd.to_datetime(request.date)
        ticker = request.ticker.upper()
        
        # Find the data for the requested ticker up to the requested date
        ticker_data = data_df.loc[ticker]
        end_idx = ticker_data.index.get_loc(end_date, method='pad')
        
        # Get the last `sequence_length` days of data
        start_idx = end_idx - sequence_length + 1
        if start_idx < 0:
            raise HTTPException(status_code=400, detail=f"Not enough historical data to make a prediction for this date. Need {sequence_length} days.")

        # Select the feature columns and scale them
        feature_cols = x_scaler.feature_names_in_
        sequence_data = ticker_data.iloc[start_idx:end_idx+1][feature_cols].values
        sequence_scaled = x_scaler.transform(sequence_data)
        
        # --- Make Prediction ---
        # Convert to tensor and add a batch dimension
        input_tensor = torch.from_numpy(sequence_scaled).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
        
        # Inverse transform the prediction to get the actual price
        prediction_unscaled = y_scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
        
        logging.info(f"Prediction for {ticker} on {request.date}: ${prediction_unscaled:.2f}")
        
        return {"predicted_price": prediction_unscaled}
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Data not found for ticker '{ticker}' or date '{request.date}'")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{ticker}")
def get_history(ticker: str):
    """
    Retrieves the historical Close price for a given ticker.
    """
    data_df = cache.get("data")
    if data_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded or is unavailable.")
    
    try:
        # Data is indexed by [Ticker, Date], so we can select the ticker
        ticker_data = data_df.loc[ticker.upper()]
        
        # Prepare data for JSON response
        history = ticker_data[['Close']].reset_index()
        return history.to_dict('records')
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found in dataset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))