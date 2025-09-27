import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(X_data, y_data, sequence_length):
    """Converts a time-series dataset into sequences for LSTM training."""
    X, y = [], []
    for i in range(len(X_data) - sequence_length):
        X.append(X_data[i:(i + sequence_length)])
        y.append(y_data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data_for_lstm(data_path: Path, output_dir: Path, sequence_length: int = 60, test_start_date: str = "2023-01-01"):
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'])

    main_ticker = df['Ticker'].value_counts().idxmax()
    logging.info(f"Automatically selected ticker: {main_ticker}")
    df_ticker = df[df['Ticker'] == main_ticker].copy().set_index('Date')
    
    # --- UPDATED: Define features and target explicitly ---
    target_col = 'Close'
    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume'] + [col for col in df.columns if 'volume_' in col or 'momentum_' in col]
    features_to_scale = [f for f in features_to_scale if f in df_ticker.columns]
    
    features_df = df_ticker[features_to_scale].dropna()
    target_series = df_ticker[[target_col]].dropna()

    # Align data by index
    aligned_index = features_df.index.intersection(target_series.index)
    features_df = features_df.loc[aligned_index]
    target_series = target_series.loc[aligned_index]

    # Chronological Split
    train_features = features_df[features_df.index < test_start_date]
    test_features = features_df[features_df.index >= test_start_date]
    train_target = target_series[target_series.index < test_start_date]
    test_target = target_series[target_series.index >= test_start_date]

    # --- UPDATED: Use two separate scalers ---
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_features_scaled = x_scaler.fit_transform(train_features)
    train_target_scaled = y_scaler.fit_transform(train_target)
    
    # --- Create Sequences ---
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, sequence_length)
    
    # Create test sequences
    all_features = pd.concat([train_features, test_features])
    inputs = all_features[len(all_features) - len(test_features) - sequence_length:].values
    inputs_scaled = x_scaler.transform(inputs)
    
    all_target = pd.concat([train_target, test_target])
    target_inputs = all_target[len(all_target) - len(test_target) - sequence_length:].values
    
    X_test, y_test = create_sequences(inputs_scaled, target_inputs, sequence_length)
    
    logging.info(f"Training sequences shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Testing sequences shape: X={X_test.shape}, y={y_test.shape}")
    
    # --- Save Outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    joblib.dump(x_scaler, output_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "y_scaler.joblib")
    
    logging.info(f"Sequence data and scalers saved to {output_dir}")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = Path("data/processed/processed_market_data.csv")
    SEQUENCE_DATA_DIR = Path("data/sequences")
    prepare_data_for_lstm(data_path=PROCESSED_DATA_PATH, output_dir=SEQUENCE_DATA_DIR)