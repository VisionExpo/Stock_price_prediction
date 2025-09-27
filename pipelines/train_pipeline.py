import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path: Path, model_output_path: Path):
    """
    Loads processed data, trains a RandomForestRegressor model, and saves it.

    Args:
        data_path (Path): Path to the processed data CSV file.
        model_output_path (Path): Path to save the trained model file.
    """
    logging.info(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    # ----- Feature Engineering : Create Target Variable-----
    # We want to predict the next day's 'Close' price
    logging.info("Creating target variable...")
    df['Target'] = df.groupby('Ticker')['Close'].shift(-1)

    # Drop rows where the target is NaN (last day for each ticker)
    df = df.dropna(subset=['Target'])

    # Define features (X) and target (y)
    features_to_exclude = ['Date', 'Ticker', 'Target', 'Close', 'Adj Close']
    features = [col for col in df.columns if col not in features_to_exclude]

    X = df[features]
    y = df['Target']

    logging.info(f"Features for training: {X.columns.tolist()}")
    logging.info(f"Data shape for training: X={X.shape}, y={y.shape}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- Model Training -----
    logging.info("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Model training completed. Test Set MSE: {mse:.4f}")

    # ----- Save the trained model -----
    logging.info(f"Saving trained model to {model_output_path}...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    logging.info("Model saved successfully.")



if __name__ == "__main__":
    # Define paths
    PROCESSED_DATA_PATH = Path("data/processed/processed_market_data.csv")
    MODEL_PATH = Path("models/random_forest_honest_baseline.joblib")

    # Execute the training function
    train_model(data_path=PROCESSED_DATA_PATH, model_output_path=MODEL_PATH)