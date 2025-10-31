from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from backend.services import predictor
import logging
import pandas as pd
from datetime import date

router = APIRouter()

class ScreenerRequest(BaseModel):
    tickers: List[str]
    date: str

@router.post("/screener")
def run_screener(request: ScreenerRequest):
    """
    Runs the prediction model on a list of tickers for a given date.
    """
    results = []
    # Use today's date for prediction
    prediction_date = date(2025, 9, 26)

    for ticker in request.tickers:
        try:
            # Reuse the existing prediction logic
            predicted_price = predictor.make_prediction(ticker=ticker, date_str=request.date)
            
            # Get last known price for comparison
            data_df = predictor.cache.get("data")
            if data_df is not None:
                ticker_data = data_df[data_df['Ticker'] == ticker.upper()]
                last_price = ticker_data[ticker_data['Date'] < pd.to_datetime(request.date)].iloc[-1]['Close']
                change_pct = ((predicted_price - last_price) / last_price) * 100 if last_price > 0 else 0
            else:
                last_price = "N/A"
                change_pct = "N/A"

            results.append({
                "Ticker": ticker,
                "Last Close": last_price,
                "Predicted Close": predicted_price,
                "Predicted Change (%)": change_pct
            })
        except Exception as e:
            logging.warning(f"Screener failed for ticker {ticker}: {e}")
            results.append({
                "Ticker": ticker,
                "Last Close": "N/A",
                "Predicted Close": "Error",
                "Predicted Change (%)": "N/A"
            })
    return results