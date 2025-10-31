from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.predictor import cache  # Import cache to get resources
import pandas as pd
import torch
import logging

router = APIRouter()

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

@router.post("/backtest")
def run_backtest(request: BacktestRequest):
    # This logic can also be moved to a 'backtester.py' service later
    # For now, we'll keep it here for simplicity.
    model = cache.get("model"); data_df = cache.get("data"); x_scaler = cache.get("x_scaler"); y_scaler = cache.get("y_scaler"); device = cache.get("device")
    if not all([model, data_df is not None, x_scaler, y_scaler]):
        raise HTTPException(status_code=500, detail="Server resources not loaded.")
    # ... (Paste the full backtesting for loop and logic from your old main.py here) ...
    try:
        ticker_data = data_df[data_df['Ticker'] == request.ticker.upper()]
        backtest_period = ticker_data[(ticker_data['Date'] >= pd.to_datetime(request.start_date)) & (ticker_data['Date'] <= pd.to_datetime(request.end_date))]
        if len(backtest_period) < 61: raise HTTPException(status_code=400, detail="Date range too short.")
        portfolio_value = 10000.0; equity_curve = []; feature_cols = x_scaler.feature_names_in_
        for i in range(60, len(backtest_period)):
            sequence_df = backtest_period.iloc[i-60:i]
            current_price = sequence_df.iloc[-1]['Close']
            sequence_scaled = x_scaler.transform(sequence_df[feature_cols])
            input_tensor = torch.from_numpy(sequence_scaled).float().unsqueeze(0).to(device)
            with torch.no_grad():
                prediction_scaled = model(input_tensor)
            predicted_price = y_scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
            actual_next_price = backtest_period.iloc[i]['Close']
            if predicted_price > current_price:
                daily_return = (actual_next_price - current_price) / current_price
                portfolio_value *= (1 + daily_return)
            equity_curve.append({'Date': backtest_period.iloc[i]['Date'].strftime('%Y-%m-%d'), 'Portfolio Value': portfolio_value})
        total_return = (portfolio_value / 10000.0 - 1) * 100
        return {"total_return_pct": total_return, "final_portfolio_value": portfolio_value, "equity_curve": equity_curve}
    except Exception as e:
        logging.error(f"Error during backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))