from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.services import predictor
from backend.routers import predict, backtest, drift, screener
import logging

logging.basicConfig(level=logging.INFO)

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all ML resources on startup
    predictor.load_resources()
    yield
    # Clean up resources on shutdown
    predictor.cache.clear()

app = FastAPI(
    title="Stock Market AI API (Refactored)",
    lifespan=lifespan
)

# --- Include Routers ---
# This brings in all the endpoints from your other files
app.include_router(predict.router)
app.include_router(backtest.router)
app.include_router(drift.router)
app.include_router(screener.router)

# --- Other General Endpoints ---
from backend.services.predictor import cache
from fastapi import HTTPException

@app.get("/tickers")
def get_tickers():
    data_df = cache.get("data")
    if data_df is None: raise HTTPException(status_code=500, detail="Data not loaded.")
    return data_df['Ticker'].unique().tolist()

@app.get("/history/{ticker}")
def get_history(ticker: str):
    data_df = cache.get("data")
    if data_df is None: raise HTTPException(status_code=500, detail="Data not loaded.")
    ticker_data = data_df[data_df['Ticker'] == ticker.upper()]
    if ticker_data.empty: raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
    return ticker_data[['Date', 'Close']].to_dict('records')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market AI API!"}