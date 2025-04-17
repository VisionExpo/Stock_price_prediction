import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def retrieve_data(ticker, days=365):
    """
    Retrieve stock data using yfinance without saving locally
    
    Args:
        ticker (str): Stock symbol
        days (int): Number of days of historical data to retrieve
    
    Returns:
        pandas.Series: Stock closing prices
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df['Close']  # Return only the closing prices

if __name__ == "__main__":
    retrieve_data('AAPL')
