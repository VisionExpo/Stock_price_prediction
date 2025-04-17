"""
Main script to train the Stock Price Prediction model
"""

import argparse
import os
from src.data_loader import prepare_data
from src.model import train_model
from src import config

def main(args):
    """
    Main function to train the model

    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("Loading and preparing data...")
    X_train, Y_train, X_test, Y_test, scaler = prepare_data(
        symbol=args.symbol,
        look_back=args.look_back,
        train_size=args.train_test_split
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("Training model...")
    model, history = train_model(
        X_train, Y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=args.verbose
    )

    model_path = os.path.join(config.MODEL_PATH, 'lstm_model.h5')
    print(f"Model trained and saved to {model_path}")

    return model, history, X_train, Y_train, X_test, Y_test, scaler

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
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stock Price Prediction LSTM model")
    add_arguments(parser)
    args = parser.parse_args()

    main(args)
