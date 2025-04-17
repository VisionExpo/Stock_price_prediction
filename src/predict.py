"""
Make predictions with the trained Stock Price Prediction model
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import load_stock_data
from src.model import load_trained_model, predict, inverse_transform
from src.utils import create_dataset
from src import config

def prepare_prediction_data(data, look_back, scaler=None):
    """
    Prepare data for prediction

    Args:
        data (pandas.Series): Stock price data
        look_back (int): Number of previous time steps to use as input features
        scaler (sklearn.preprocessing.MinMaxScaler, optional): Scaler to use. If None, a new one will be created

    Returns:
        tuple: (X, scaler)
    """
    # Convert to numpy array
    dataset = data.values.reshape(-1, 1)

    # Normalize the dataset
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
    else:
        dataset = scaler.transform(dataset)

    # Use the last 'look_back' data points as input
    X = dataset[-look_back:].reshape(1, look_back, 1)

    return X, scaler

def predict_future(model, last_data, scaler, days=30, look_back=None):
    """
    Predict future stock prices

    Args:
        model (tensorflow.keras.models.Sequential): Trained model
        last_data (numpy.ndarray): Last 'look_back' data points
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to normalize data
        days (int): Number of days to predict
        look_back (int, optional): Number of previous time steps to use as input features

    Returns:
        numpy.ndarray: Predicted prices
    """
    if look_back is None:
        look_back = config.LOOK_BACK

    # Make a copy of the last data
    curr_data = last_data.copy()
    future_predictions = []

    for _ in range(days):
        # Predict the next price
        prediction = model.predict(curr_data)
        future_predictions.append(prediction[0, 0])

        # Update the input data for the next prediction
        curr_data = np.append(curr_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    # Inverse transform to get actual prices
    dummy = np.zeros((len(future_predictions), 1))
    dummy[:, 0] = future_predictions
    future_predictions = scaler.inverse_transform(dummy)[:, 0]

    return future_predictions

def main(args):
    """
    Main function to make predictions

    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("Loading stock data...")
    df = load_stock_data(args.symbol)

    # Extract the 'close' price
    if 'close' in df.columns:
        data = df['close']
    else:
        # If the dataframe has a multi-level index, reset it
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        data = df['close']

    print("Loading trained model...")
    model = load_trained_model(args.model_path)

    print("Preparing data for prediction...")
    X, scaler = prepare_prediction_data(data, args.look_back)

    if args.future_days > 0:
        print(f"Predicting future prices for {args.future_days} days...")
        future_predictions = predict_future(model, X, scaler, args.future_days, args.look_back)

        # Create dates for future predictions
        last_date = pd.to_datetime(df['date'].iloc[-1] if 'date' in df.columns else df.index[-1])
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(args.future_days)]

        # Create a dataframe with the predictions
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': future_predictions
        })

        print("\nFuture Price Predictions:")
        print(future_df)

        # Create results directory if it doesn't exist
        if not os.path.exists(config.RESULTS_PATH):
            os.makedirs(config.RESULTS_PATH)

        # Save predictions to CSV
        predictions_file = os.path.join(config.RESULTS_PATH, f'{args.symbol}_future_predictions.csv')
        future_df.to_csv(predictions_file, index=False)
        print(f"Predictions saved to {predictions_file}")

        # Plot predictions
        if args.plot:
            plt.figure(figsize=(12, 6))

            # Plot historical data
            plt.plot(data.values[-100:], label='Historical')

            # Plot future predictions
            plt.plot(range(len(data)-1, len(data)+args.future_days-1), future_predictions, label='Predicted')

            plt.title(f"{args.symbol} Stock Price Prediction")
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()

            plot_file = os.path.join(config.RESULTS_PATH, f'{args.symbol}_future_predictions.png')
            plt.savefig(plot_file)
            plt.show()

    return future_predictions if args.future_days > 0 else None

def add_arguments(parser):
    """
    Add command line arguments

    Args:
        parser (argparse.ArgumentParser): Argument parser
    """
    # Data parameters
    parser.add_argument("--symbol", type=str, default=config.STOCK_SYMBOL,
                        help=f"Stock symbol (default: {config.STOCK_SYMBOL})")

    # Model parameters
    parser.add_argument("--look-back", type=int, default=config.LOOK_BACK,
                        help=f"Number of previous time steps to use as input (default: {config.LOOK_BACK})")
    parser.add_argument("--model-path", type=str, default=os.path.join(config.MODEL_PATH, "lstm_model.h5"),
                        help=f"Path to trained model (default: {os.path.join(config.MODEL_PATH, 'lstm_model.h5')})")

    # Prediction parameters
    parser.add_argument("--future-days", type=int, default=30,
                        help="Number of days to predict into the future (default: 30)")

    # Output parameters
    parser.add_argument("--plot", action="store_true",
                        help="Plot predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with Stock Price Prediction LSTM model")
    add_arguments(parser)
    args = parser.parse_args()

    main(args)
