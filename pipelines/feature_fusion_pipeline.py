import pandas as pd
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fuse_features(market_data_path: Path, sentiment_data_path: Path, output_path: Path):
    """
    Merges processed market data with processed sentiment data.
    """
    logging.info("Starting feature fusion process...")
    
    try:
        # Load the two datasets
        market_df = pd.read_csv(market_data_path, parse_dates=['Date'])
        sentiment_df = pd.read_csv(sentiment_data_path, parse_dates=['Date'])
        logging.info("Market and sentiment data loaded successfully.")

        # Perform a left merge to keep all market data and add sentiment where available
        fused_df = pd.merge(market_df, sentiment_df, on='Date', how='left')
        
        # --- Handle Missing Sentiment ---
        # For dates with no news, the sentiment_score will be NaN.
        # We'll fill these with 0, representing neutral sentiment.
        fused_df['sentiment_score'].fillna(0, inplace=True)
        logging.info("Sentiment scores merged and missing values filled.")
        
        # Save the final fused dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fused_df.to_csv(output_path, index=False)
        
        logging.info(f"Feature fusion complete. Final dataset shape: {fused_df.shape}")
        logging.info(f"Fused data saved to: {output_path}")

    except FileNotFoundError as e:
        logging.error(f"File not found during fusion: {e}. Please ensure previous pipelines have run.")
    except Exception as e:
        logging.error(f"An error occurred during feature fusion: {e}")


if __name__ == "__main__":
    # Define paths
    MARKET_DATA_PATH = Path("data/processed/processed_market_data.csv")
    SENTIMENT_DATA_PATH = Path("data/processed/processed_sentiment_scores.csv")
    FUSED_DATA_PATH = Path("data/processed/final_fused_data.csv")
    
    # Execute the fusion function
    fuse_features(
        market_data_path=MARKET_DATA_PATH,
        sentiment_data_path=SENTIMENT_DATA_PATH,
        output_path=FUSED_DATA_PATH
    )