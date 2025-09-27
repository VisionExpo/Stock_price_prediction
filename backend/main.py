from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import joblib
import pandas as pd
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model & App Cache ---
cache = {}

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and data during startup
    logging.info("Application startup: Loading model and data...")

    # Load model
    model_path = Path("models/random_forest_v1.joblib")
    if model_path.exists():
        cache["model"] = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    else:
        cache["model"] = None
        logging.error(f"Model file not found at {model_path}")

    # Load the processed dataset as well
    data_path = Path("data/processed/processed_market_data.csv")
    if data_path.exists():
        cache["data"] = pd.read_csv(data_path, parse_dates=['Date'])
        logging.info(f"Data loaded successfully. Shape: {cache['data'].shape}")
    else:
        cache["data"] = None
        logging.error(f"Data file not found at {data_path}")
        
    yield
    
    # Clean up during shutdown
    logging.info("Application shutdown: Clearing cache...")
    cache.clear()

app = FastAPI(
    title="Stock Market AI API",
    description="API for predicting stock prices using an ML model.",
    lifespan=lifespan
)

# --- Pydantic Models for Data Validation ---
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
    model = cache.get("model")
    data_df = cache.get("data")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or is unavailable.")
    if data_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded or is unavailable.")
    
    try:
        # Convert request date string to datetime to match DataFrame's date format
        request_date = pd.to_datetime(request.date).date()
        
        # Find the specific row in the data for the requested ticker and date
        features_row = data_df[
            (data_df['Ticker'] == request.ticker.upper()) & 
            (data_df['Date'].dt.date == request_date)
        ]

        if features_row.empty:
            raise HTTPException(status_code=404, detail=f"Data not found for ticker '{request.ticker}' on date '{request.date}'")

        # Prepare the features for the model (exclude non-feature columns)
        features_to_exclude = ['Date', 'Ticker', 'target']
        features = [col for col in features_row.columns if col not in features_to_exclude]
        
        features_df_for_prediction = features_row[features]
        
        logging.info(f"Found data for {request.ticker} on {request.date}. Predicting...")

        # Make a prediction
        prediction = model.predict(features_df_for_prediction)[0]
        logging.info(f"Prediction result: {prediction}")
        
        return {"predicted_price": prediction}
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))