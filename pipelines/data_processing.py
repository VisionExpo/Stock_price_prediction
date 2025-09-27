import pandas as pd
from pathlib import Path
import logging
import ta # Technical Analysis library

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_path: Path, output_path: Path):
    """
    Preprocesses the stock data by handling missing values and adding technical indicators.

    Args:
        input_path (Path): The Path object where the input CSV is located.
        output_path (Path): The Path object where the preprocessed CSV will be saved.
    """
    logging.info(f"Starting data preprocessing from {input_path}")

    try:
        # Read the data with MultiIndex header (for multiple tickers)
        df= pd.read_csv(input_path, header=[0,1], index_col=0, parse_dates=True)

        # Stack the columns to make a single column, easier to work with
        df = df.stack(level=0).reset_index().rename(columns={'level_1': 'Ticker'})

        logging.info("Cleaning data: forward filling missing values...")
        # Forward fill missing values, common in time series data
        df = df.sort_values(by=['Ticker', 'Date'])
        df = df.set_index(['Ticker', 'Date']).groupby(level='Ticker').ffill().reset_index()

        logging.info("Adding all technical analysis features...")

        # Use the 'ta' library to add all technical indicators
        df = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True    # Fill NaN values if created by indicators
        )

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the preprocessed data
        df.to_csv(output_path, index=False)

        logging.info(f"Successfully processed data. Shape: {df.shape}")
        logging.info(f"Processed data saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")


if __name__ == "__main__":
    #  Define input and output file paths
    RAW_DATA_PATH = Path("data/raw/raw_market_data.csv")
    PROCESSED_DATA_PATH = Path("data/processed/processed_market_data.csv")

    # Call the preprocessing function
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)