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

    Raises:
        ValueError: If no data is found for the ticker or if there's an error
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Retrieving data for {ticker} from {start_date} to {end_date}")

        # Try to download the data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Check if we got any data
        if df.empty:
            logger.error(f"No data found for ticker {ticker}")
            raise ValueError(f"No data found for ticker {ticker}. Please check if the symbol is correct.")

        # Check if we have closing prices
        if 'Close' not in df.columns:
            logger.error(f"No closing price data found for ticker {ticker}")
            raise ValueError(f"No closing price data found for ticker {ticker}.")

        logger.info(f"Successfully retrieved {len(df)} data points for {ticker}")
        return df['Close']  # Return only the closing prices

    except Exception as e:
        logger.error(f"Error retrieving data for {ticker}: {str(e)}")
        # If it's already a ValueError, re-raise it
        if isinstance(e, ValueError):
            raise
        # Otherwise, wrap it in a ValueError with a more user-friendly message
        raise ValueError(f"Error retrieving data for {ticker}. Please check if the symbol is correct and try again. Details: {str(e)}")

if __name__ == "__main__":
    retrieve_data('AAPL')
