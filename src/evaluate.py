"""
Evaluate the Stock Price Prediction model
"""

import argparse
import os
import numpy as np
from src.data_loader import prepare_data
from src.model import load_trained_model, predict, inverse_transform
from src.utils import plot_predictions, calculate_metrics, save_results
from src import config

def evaluate_model(model, X_test, Y_test, scaler):
    """
    Evaluate the model on test data

    Args:
        model (tensorflow.keras.models.Sequential): Trained model
        X_test (numpy.ndarray): Test input data
        Y_test (numpy.ndarray): Test target data
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to normalize data

    Returns:
        tuple: (actual_prices, predicted_prices, metrics)
    """
    # Make predictions
    predictions = predict(model, X_test)

    # Inverse transform to get actual prices
    actual_prices = inverse_transform(Y_test, scaler)
    predicted_prices = inverse_transform(predictions, scaler)

    # Calculate metrics
    metrics = calculate_metrics(actual_prices, predicted_prices)

    return actual_prices, predicted_prices, metrics

def main(args):
    """
    Main function to evaluate the model

    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("Loading and preparing data...")
    X_train, Y_train, X_test, Y_test, scaler = prepare_data(
        symbol=args.symbol,
        look_back=args.look_back,
        train_size=args.train_test_split
    )

    print("Loading trained model...")
    model = load_trained_model(args.model_path)

    print("Evaluating model...")
    actual_prices, predicted_prices, metrics = evaluate_model(model, X_test, Y_test, scaler)

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics
    save_results(metrics, args.metrics_file)

    # Plot predictions
    if args.plot:
        plot_predictions(actual_prices, predicted_prices, title=f"{args.symbol} Stock Price Prediction")

    return actual_prices, predicted_prices, metrics

def add_arguments(parser):
    """
    Add command line arguments

    Args:
        parser (argparse.ArgumentParser): Argument parser
    """
    # Data parameters
    parser.add_argument("--symbol", type=str, default=config.STOCK_SYMBOL,
                        help=f"Stock symbol (default: {config.STOCK_SYMBOL})")
    parser.add_argument("--train-test-split", type=float, default=config.TRAIN_TEST_SPLIT,
                        help=f"Train-test split ratio (default: {config.TRAIN_TEST_SPLIT})")

    # Model parameters
    parser.add_argument("--look-back", type=int, default=config.LOOK_BACK,
                        help=f"Number of previous time steps to use as input (default: {config.LOOK_BACK})")
    parser.add_argument("--model-path", type=str, default=os.path.join(config.MODEL_PATH, "lstm_model.h5"),
                        help=f"Path to trained model (default: {os.path.join(config.MODEL_PATH, 'lstm_model.h5')})")

    # Output parameters
    parser.add_argument("--metrics-file", type=str, default="metrics.csv",
                        help="File to save metrics (default: metrics.csv)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stock Price Prediction LSTM model")
    add_arguments(parser)
    args = parser.parse_args()

    main(args)
