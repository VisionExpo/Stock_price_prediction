"""
Performance tests for the LSTM model
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import prepare_data
from src.model import train_model, predict, inverse_transform
from src.utils import calculate_metrics, plot_predictions
from src import config

def test_model_performance(symbol='AAPL', look_back=60, epochs=100, batch_size=32):
    """
    Test model performance on real stock data

    Args:
        symbol (str): Stock symbol
        look_back (int): Number of previous time steps to use as input
        epochs (int): Number of training epochs
        batch_size (int): Batch size

    Returns:
        dict: Performance metrics
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join('tests', 'performance', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Start timer
    start_time = time.time()

    # Load data
    print(f"Testing model performance for {symbol}...")

    # Try to load data from Tiingo API
    try:
        import os
        import pandas_datareader as pdr
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        # Get API key
        api_key = os.getenv("TIINGO_API_KEY")
        if not api_key:
            raise ValueError("TIINGO_API_KEY not found in environment variables")

        # Fetch data
        df = pdr.get_data_tiingo(symbol, api_key=api_key)

        # Reset index if multi-level
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
    except Exception as e:
        print(f"Error loading data from Tiingo API: {e}")
        # Generate synthetic data if loading fails
        print("Using synthetic data instead...")
        dates = pd.date_range(start='2020-01-01', periods=500)
        close = np.sin(np.linspace(0, 10, 500)) * 50 + 100 + np.random.randn(500) * 5
        df = pd.DataFrame({'date': dates, 'close': close})

    # Prepare data
    X_train, Y_train, X_test, Y_test, scaler = prepare_data(
        df=df,
        look_back=look_back,
        train_size=0.8
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Train model
    print("Training model...")
    model, history = train_model(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Make predictions
    print("Making predictions...")
    predictions = predict(model, X_test)

    # Inverse transform
    actual_prices = inverse_transform(Y_test, scaler)
    predicted_prices = inverse_transform(predictions, scaler)

    # Calculate metrics
    metrics = calculate_metrics(actual_prices, predicted_prices)

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df['symbol'] = symbol
    metrics_df['look_back'] = look_back
    metrics_df['epochs'] = epochs
    metrics_df['batch_size'] = batch_size
    metrics_df['training_time'] = training_time
    metrics_df['train_samples'] = len(X_train)
    metrics_df['test_samples'] = len(X_test)

    metrics_file = os.path.join(results_dir, f'{symbol}_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    history_file = os.path.join(results_dir, f'{symbol}_training_history.png')
    plt.savefig(history_file)
    print(f"Training history plot saved to {history_file}")

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual')
    plt.plot(predicted_prices, label='Predicted')
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    predictions_file = os.path.join(results_dir, f'{symbol}_predictions.png')
    plt.savefig(predictions_file)
    print(f"Predictions plot saved to {predictions_file}")

    return metrics

def compare_models(symbols=['AAPL', 'MSFT', 'GOOGL'], look_backs=[30, 60, 90]):
    """
    Compare model performance across different stocks and parameters

    Args:
        symbols (list): List of stock symbols
        look_backs (list): List of look-back periods to test

    Returns:
        pandas.DataFrame: Comparison results
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join('tests', 'performance', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize results dataframe
    results = []

    # Test each combination
    for symbol in symbols:
        for look_back in look_backs:
            print(f"\n{'='*50}")
            print(f"Testing {symbol} with look_back={look_back}")
            print(f"{'='*50}\n")

            # Test model performance
            metrics = test_model_performance(
                symbol=symbol,
                look_back=look_back,
                epochs=50,  # Reduced for testing multiple models
                batch_size=32
            )

            # Add to results
            metrics['symbol'] = symbol
            metrics['look_back'] = look_back
            results.append(metrics)

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Save comparison results
    comparison_file = os.path.join(results_dir, 'model_comparison.csv')
    results_df.to_csv(comparison_file, index=False)
    print(f"\nComparison results saved to {comparison_file}")

    # Create comparison plots
    plt.figure(figsize=(12, 8))

    # Plot MSE by symbol and look_back
    for symbol in symbols:
        symbol_results = results_df[results_df['symbol'] == symbol]
        plt.plot(symbol_results['look_back'], symbol_results['MSE'], marker='o', label=symbol)

    plt.title('MSE by Look-Back Period')
    plt.xlabel('Look-Back Period')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    mse_plot_file = os.path.join(results_dir, 'mse_comparison.png')
    plt.savefig(mse_plot_file)
    print(f"MSE comparison plot saved to {mse_plot_file}")

    return results_df

if __name__ == "__main__":
    # Test single model performance
    test_model_performance(symbol='AAPL', look_back=60, epochs=100, batch_size=32)

    # Uncomment to compare multiple models
    # compare_models(symbols=['AAPL', 'MSFT', 'GOOGL'], look_backs=[30, 60, 90])
