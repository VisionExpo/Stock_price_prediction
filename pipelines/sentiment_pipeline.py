import os
import requests
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_news_data(api_key: str, ticker: str, start_date: str, end_date: str, output_path: Path):
    """
    Fetches news headlines for a given stock ticker and saves them to a CSV.
    """
    logging.info(f"Fetching news for {ticker} from {start_date} to {end_date}...")
    
    # The NewsAPI endpoint for everything
    url = 'https://newsapi.org/v2/everything'
    
    # Construct a query. Search for the ticker AND stock-related keywords for relevance.
    query = f'({ticker} OR "{ticker} stock")'
    
    params = {
        'q': query,
        'from': start_date,
        'to': end_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'pageSize': 100 # Max results per page
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        articles = response.json().get('articles', [])
        
        if not articles:
            logging.warning(f"No articles found for {ticker} in the specified date range.")
            return

        # Create a DataFrame from the articles
        df = pd.DataFrame(articles)
        df = df[['publishedAt', 'title', 'source']]
        df['source'] = df['source'].apply(lambda x: x['name']) # Extract source name
        df.rename(columns={'publishedAt': 'Date', 'title': 'Headline'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date # Keep only the date part
        
        # Save the data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Successfully fetched and saved {len(df)} articles to {output_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # --- Configuration ---
    API_KEY = os.getenv("NEWS_API_KEY")
    TICKER = "AAPL" # We can make this dynamic later
    START_DATE = "2025-08-28" # NewsAPI free tier has limitations on historical data
    END_DATE = "2025-09-27"
    OUTPUT_FILE = Path("data/raw/raw_news_data.csv")
    
    if not API_KEY:
        logging.error("NEWS_API_KEY not found in environment variables. Please check your .env file.")
    else:
        fetch_news_data(
            api_key=API_KEY,
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            output_path=OUTPUT_FILE
        )