"""
Entry point for the Stock Price Prediction application
"""

import argparse
import sys
from src import main, train, evaluate, predict

def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Stock Price Prediction using LSTM")

    # Command selection
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Main command
    main_parser = subparsers.add_parser('main', help='Run the complete pipeline')
    main.add_arguments(main_parser)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train.add_arguments(train_parser)

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    evaluate.add_arguments(evaluate_parser)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict.add_arguments(predict_parser)

    return parser.parse_args()

def main_func():
    """
    Main function to run the application
    """
    args = parse_args()

    if args.command == 'main':
        main.main(args)
    elif args.command == 'train':
        train.main(args)
    elif args.command == 'evaluate':
        evaluate.main(args)
    elif args.command == 'predict':
        predict.main(args)
    else:
        print("Please specify a command: main, train, evaluate, or predict")
        sys.exit(1)

if __name__ == "__main__":
    main_func()
