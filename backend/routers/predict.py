from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services import predictor
import logging

router = APIRouter()

class PredictionRequest(BaseModel):
    ticker: str
    date: str

class PredictionResponse(BaseModel):
    predicted_price: float

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = predictor.make_prediction(ticker=request.ticker, date_str=request.date)
        return {"predicted_price": prediction}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")