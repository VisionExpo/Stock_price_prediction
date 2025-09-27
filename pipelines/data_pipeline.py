import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_stock_data(ticker: list, start_date: str, end_date: str, output_path: Path):
    """
    Downloads historical stock data for a list of tickers and saves it to a CSV file.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'GOOGL']).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        output_path (Path): The Path object where the output CSV will be saved.
    """
    logging.info(f"Starting data ingestion for tickers: {ticker}")

    try:
        # Download historical data from yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
        
        if stock_data.empty:
            logging.warning("No data downloaded. Please check the ticker symbols and date range.")
            return
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the data to a CSV file
        stock_data.to_csv(output_path)

        logging.info(f"Successfully downloaded {len(stock_data)} rows of data.")
        logging.info(f"Data saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during data ingestion: {e}")


if __name__ == "__main__":
    TICKERS_TO_INGEST = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    START_DATE = "2015-01-01"
    END_DATE = "2025-09-25"

    OUTPUT_FILE = Path("data/raw/raw_market_data.csv")


    ingest_stock_data(
        ticker=TICKERS_TO_INGEST,
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=OUTPUT_FILE
        )