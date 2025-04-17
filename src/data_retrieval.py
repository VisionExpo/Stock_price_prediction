import pandas as pd
import os
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta

def retrieve_data(ticker):
    load_dotenv()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year of data
    
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f'{ticker}.csv')
    return df['Close']  # Return only the closing prices

if __name__ == "__main__":
    retrieve_data('AAPL')
