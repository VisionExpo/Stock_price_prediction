from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any # Import 'Any'
from pathlib import Path
import joblib
import pandas as pd
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model & App Cache ---
model_cache = {}

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model during startup
    logging.info("Application startup: Loading model...")
    model_path = Path("model/random_forest_v1.joblib")
    if model_path.exists():
        model_cache["model"] = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    else:
        model_cache["model"] = None
        logging.error(f"Model file not found at {model_path}")
    yield
    # Clean up during shutdown
    logging.info("Application shutdown: Clearing model cache...")
    model_cache.clear()

app = FastAPI(
    title="Stock Market AI API",
    description="API for predicting stock prices using an ML model.",
    lifespan=lifespan
)

# --- Pydantic Models for Data Validation ---
class PredictionRequest(BaseModel):
    # UPDATED: Allow any value type, not just float, to accept strings like 'Ticker' and 'Date'
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    predicted_price: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market AI API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = model_cache.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or is unavailable.")

    # --- ADDED: Filter out non-numeric features before prediction ---
    # The model was not trained on 'Ticker' or 'Date', so we must remove them.
    features_for_prediction = {
        key: value for key, value in request.features.items() 
        if key not in ['Ticker', 'Date']
    }
    
    try:
        features_df = pd.DataFrame([features_for_prediction])
        logging.info(f"DataFrame created for prediction with {len(features_df.columns)} columns.")
        
        # Make a prediction
        prediction = model.predict(features_df)[0]
        logging.info(f"Prediction result: {prediction}")
        
        return {"predicted_price": prediction}
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))