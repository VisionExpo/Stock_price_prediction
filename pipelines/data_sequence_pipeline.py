import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(data, sequence_length):
    """Converts a time-series dataset into sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :]) # Features
        y.append(data[i + sequence_length, 3])      # Target is the 'Close' price, index 3
    return np.array(X), np.array(y)

def prepare_data_for_lstm(data_path: Path, output_dir: Path, sequence_length: int = 60, test_start_date: str = "2023-01-01"):
    """
    Loads processed data, scales it, creates sequences, and saves them.
    """
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'])

    # --- NEW: Dynamically select the ticker with the most data ---
    if df.empty:
        logging.error("The processed data file is empty. Aborting.")
        return
    main_ticker = df['Ticker'].value_counts().idxmax()
    logging.info(f"Automatically selected ticker with the most data: {main_ticker}")
    
    df_ticker = df[df['Ticker'] == main_ticker].copy().set_index('Date')
    
    # --- Data Scaling ---
    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume'] + [col for col in df.columns if 'volume_' in col or 'momentum_' in col]
    # Ensure all selected features exist in the DataFrame
    features_to_scale = [f for f in features_to_scale if f in df_ticker.columns]
    
    features_df = df_ticker[features_to_scale].dropna()
    
    # Chronological Split
    train_df = features_df[features_df.index < test_start_date]
    test_df = features_df[features_df.index >= test_start_date]

    if train_df.empty or test_df.empty:
        logging.error(f"Not enough data for ticker {main_ticker} to create a train/test split around {test_start_date}. Aborting.")
        return

    # Scale the data based on the training set ONLY to avoid data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(train_df)
    
    # --- Create Training Sequences ---
    X_train, y_train = create_sequences(training_set_scaled, sequence_length)
    
    # --- Create Testing Sequences ---
    # We need the tail of the training data to form the initial sequences for the test set
    total_inputs = pd.concat((train_df, test_df), axis=0)
    inputs = total_inputs[len(total_inputs) - len(test_df) - sequence_length:].values
    scaled_inputs = scaler.transform(inputs)
    X_test, y_test = create_sequences(scaled_inputs, sequence_length)
    
    logging.info(f"Training sequences shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Testing sequences shape: X={X_test.shape}, y={y_test.shape}")
    
    # --- Save Outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    logging.info(f"Sequence data and scaler saved to {output_dir}")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = Path("data/processed/processed_market_data.csv")
    SEQUENCE_DATA_DIR = Path("data/sequences")
    
    prepare_data_for_lstm(data_path=PROCESSED_DATA_PATH, output_dir=SEQUENCE_DATA_DIR)