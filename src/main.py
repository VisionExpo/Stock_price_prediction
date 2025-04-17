"""
Main script to run the Stock Price Prediction pipeline
"""

import argparse
import os
from src.data_loader import prepare_data
from src.model import train_model, load_trained_model, predict, inverse_transform
from src.utils import plot_predictions, calculate_metrics, save_results
from src import config

def main(args):
    """
    Main function to run the pipeline

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Create necessary directories
    for directory in [config.DATA_PATH, config.MODEL_PATH, config.RESULTS_PATH]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    print("Loading and preparing data...")
    X_train, Y_train, X_test, Y_test, scaler = prepare_data(
        symbol=args.symbol,
        look_back=args.look_back,
        train_size=args.train_test_split
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model_path = os.path.join(config.MODEL_PATH, 'lstm_model.h5')
    if args.train or not os.path.exists(model_path):
        print("Training model...")
        model, history = train_model(
            X_train, Y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            verbose=args.verbose
        )
        print(f"Model trained and saved to {model_path}")
    else:
        print("Loading trained model...")
        model = load_trained_model(model_path)

    if args.evaluate:
        print("Evaluating model...")
        # Make predictions
        test_predictions = predict(model, X_test)

        # Inverse transform to get actual prices
        actual_prices = inverse_transform(Y_test, scaler)
        predicted_prices = inverse_transform(test_predictions, scaler)

        # Calculate metrics
        metrics = calculate_metrics(actual_prices, predicted_prices)

        # Print metrics
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics
        save_results(metrics, args.metrics_file)

        # Plot predictions
        if args.plot:
            plot_predictions(actual_prices, predicted_prices, title=f"{args.symbol} Stock Price Prediction")

    return model, X_train, Y_train, X_test, Y_test, scaler

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
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Number of epochs (default: {config.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--validation-split", type=float, default=config.VALIDATION_SPLIT,
                        help=f"Validation split ratio (default: {config.VALIDATION_SPLIT})")

    # Pipeline control
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model")

    # Output parameters
    parser.add_argument("--metrics-file", type=str, default="metrics.csv",
                        help="File to save metrics (default: metrics.csv)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot predictions")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction using LSTM")
    add_arguments(parser)
    args = parser.parse_args()

    # If neither train nor evaluate is specified, do both
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True

    main(args)
