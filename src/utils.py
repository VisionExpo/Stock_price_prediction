import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from functools import lru_cache
import psutil

class ModelTracker:
    def __init__(self):
        self.metrics_dir = os.environ.get('MODEL_METRICS_DIR', "model_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        print(f"ModelTracker initialized with metrics directory: {self.metrics_dir}")

    def save_metrics(self, metrics, model_params, stock_symbol):
        """Save model metrics and parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_symbol}_metrics_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "stock_symbol": stock_symbol,
            "metrics": metrics,
            "parameters": model_params
        }

        with open(os.path.join(self.metrics_dir, filename), "w") as f:
            json.dump(data, f, indent=4)

    def load_latest_metrics(self, stock_symbol):
        """Load the most recent metrics for a stock symbol"""
        files = [f for f in os.listdir(self.metrics_dir)
                if f.startswith(f"{stock_symbol}_metrics_")]
        if not files:
            return None

        latest_file = max(files)
        with open(os.path.join(self.metrics_dir, latest_file), "r") as f:
            return json.load(f)

    def get_all_metrics(self, stock_symbol):
        """Get all historical metrics for a stock symbol"""
        metrics_list = []
        for file in os.listdir(self.metrics_dir):
            if file.startswith(f"{stock_symbol}_metrics_"):
                with open(os.path.join(self.metrics_dir, file), "r") as f:
                    metrics_list.append(json.load(f))
        return metrics_list

def plot_predictions(y_true, y_pred, title="Stock Price Prediction"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    return plt.gcf()

def plot_metrics_history(metrics_list):
    """Plot historical metrics over time"""
    if not metrics_list:
        return None

    dates = [m['timestamp'] for m in metrics_list]
    mse = [m['metrics']['MSE'] for m in metrics_list]
    rmse = [m['metrics']['RMSE'] for m in metrics_list]
    mae = [m['metrics']['MAE'] for m in metrics_list]
    r2 = [m['metrics']['R2 Score'] for m in metrics_list]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(dates, mse)
    ax1.set_title('MSE History')
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(dates, rmse)
    ax2.set_title('RMSE History')
    ax2.tick_params(axis='x', rotation=45)

    ax3.plot(dates, mae)
    ax3.set_title('MAE History')
    ax3.tick_params(axis='x', rotation=45)

    ax4.plot(dates, r2)
    ax4.set_title('R² Score History')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def check_system_health():
    """Check system health metrics"""
    try:
        # Get model path from environment or use default
        model_path = os.path.join(os.environ.get('CACHE_DIR', 'cache'), 'stock_model.h5')

        health_metrics = {
            'status': 'healthy',
            'memory_usage': psutil.Process().memory_percent(),
            'cpu_usage': psutil.Process().cpu_percent(),
            'disk_usage': psutil.disk_usage('/').percent,
            'model_available': os.path.exists(model_path),
            'model_path': model_path,
            'cache_status': 'active',
            'timestamp': datetime.now().isoformat()
        }

        # Check if memory usage is too high
        if health_metrics['memory_usage'] > 90:
            health_metrics['status'] = 'warning'
            health_metrics['warning'] = 'High memory usage'

        # Check if disk space is low
        if health_metrics['disk_usage'] > 90:
            health_metrics['status'] = 'warning'
            health_metrics['warning'] = 'Low disk space'

        return health_metrics
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@lru_cache(maxsize=100)
def get_cached_stock_data(ticker: str, days: int) -> pd.DataFrame:
    """Get stock data with caching"""
    import logging
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker symbol must be a non-empty string")

    if not isinstance(days, int) or days <= 0:
        raise ValueError("Days must be a positive integer")

    cache_dir = os.environ.get('CACHE_DIR', 'cache')
    cache_file = f'{cache_dir}/{ticker}_{days}.csv'

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Using cache file: {cache_file}")

    try:
        # Check if cached data exists and is recent
        if os.path.exists(cache_file):
            try:
                cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)

                # Check if the cached data is valid
                if cached_data.empty:
                    logger.warning(f"Cached data for {ticker} is empty, fetching new data")
                else:
                    cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))

                    # Return cached data if it's less than 1 hour old
                    if cache_age < timedelta(hours=1):
                        logger.info(f"Using cached data for {ticker} (age: {cache_age})")
                        return cached_data
                    else:
                        logger.info(f"Cached data for {ticker} is too old ({cache_age}), fetching new data")
            except Exception as e:
                logger.warning(f"Error reading cached data for {ticker}: {str(e)}, fetching new data")

        # Fetch new data if cache is missing, old, or invalid
        from src.data_retrieval import retrieve_data
        logger.info(f"Fetching new data for {ticker} for the past {days} days")
        df = retrieve_data(ticker, days)

        # Verify the data is valid
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Save to cache
        try:
            df.to_csv(cache_file)
            logger.info(f"Saved data for {ticker} to cache")
        except Exception as e:
            logger.warning(f"Failed to save data to cache: {str(e)}")

        return df

    except Exception as e:
        logger.error(f"Error in get_cached_stock_data for {ticker}: {str(e)}")
        raise ValueError(f"Failed to retrieve data for {ticker}: {str(e)}")