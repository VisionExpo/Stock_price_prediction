"""
Hyperparameter tuning for the LSTM model
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from src.data_loader import prepare_data
from src.model import build_model, predict, inverse_transform
from src.utils import calculate_metrics
from src import config

def tune_hyperparameters(symbol='AAPL',
                         look_backs=[30, 60, 90],
                         lstm_units=[50, 100],
                         dropout_rates=[0.2, 0.3],
                         batch_sizes=[16, 32, 64],
                         epochs=50):
    """
    Tune hyperparameters for the LSTM model

    Args:
        symbol (str): Stock symbol
        look_backs (list): List of look-back periods to test
        lstm_units (list): List of LSTM units to test
        dropout_rates (list): List of dropout rates to test
        batch_sizes (list): List of batch sizes to test
        epochs (int): Number of training epochs

    Returns:
        pandas.DataFrame: Tuning results
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join('tests', 'performance', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print(f"Loading data for {symbol}...")

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

    # Initialize results
    results = []

    # Generate all combinations of hyperparameters
    param_combinations = list(product(look_backs, lstm_units, dropout_rates, batch_sizes))
    total_combinations = len(param_combinations)

    print(f"Testing {total_combinations} hyperparameter combinations...")

    # Test each combination
    for i, (look_back, lstm_unit, dropout_rate, batch_size) in enumerate(param_combinations):
        print(f"\nCombination {i+1}/{total_combinations}:")
        print(f"look_back={look_back}, lstm_units={lstm_unit}, dropout_rate={dropout_rate}, batch_size={batch_size}")

        # Start timer
        start_time = time.time()

        # Prepare data
        X_train, Y_train, X_test, Y_test, scaler = prepare_data(
            df=df,
            look_back=look_back,
            train_size=0.8
        )

        # Build and train model
        model = build_model(look_back=look_back, lstm_units=lstm_unit, dropout_rate=dropout_rate)

        # Set learning rate
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train model
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )

        # Calculate training time
        training_time = time.time() - start_time

        # Make predictions
        predictions = predict(model, X_test)

        # Inverse transform
        actual_prices = inverse_transform(Y_test, scaler)
        predicted_prices = inverse_transform(predictions, scaler)

        # Calculate metrics
        metrics = calculate_metrics(actual_prices, predicted_prices)

        # Add hyperparameters to metrics
        metrics['look_back'] = look_back
        metrics['lstm_units'] = lstm_unit
        metrics['dropout_rate'] = dropout_rate
        metrics['batch_size'] = batch_size
        metrics['training_time'] = training_time
        metrics['final_loss'] = history.history['loss'][-1]
        metrics['final_val_loss'] = history.history['val_loss'][-1]

        # Add to results
        results.append(metrics)

        # Print metrics
        print(f"MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, Training Time: {training_time:.2f}s")

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Save results
    tuning_file = os.path.join(results_dir, f'{symbol}_hyperparameter_tuning.csv')
    results_df.to_csv(tuning_file, index=False)
    print(f"\nHyperparameter tuning results saved to {tuning_file}")

    # Find best combination
    best_idx = results_df['MSE'].idxmin()
    best_params = results_df.iloc[best_idx]

    print("\nBest Hyperparameters:")
    print(f"look_back: {best_params['look_back']}")
    print(f"lstm_units: {best_params['lstm_units']}")
    print(f"dropout_rate: {best_params['dropout_rate']}")
    print(f"batch_size: {best_params['batch_size']}")
    print(f"MSE: {best_params['MSE']:.4f}")
    print(f"RMSE: {best_params['RMSE']:.4f}")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot MSE by look_back and lstm_units
    plt.subplot(2, 2, 1)
    for unit in lstm_units:
        unit_results = results_df[results_df['lstm_units'] == unit]
        unit_results = unit_results.groupby('look_back')['MSE'].mean().reset_index()
        plt.plot(unit_results['look_back'], unit_results['MSE'], marker='o', label=f'Units: {unit}')

    plt.title('MSE by Look-Back Period and LSTM Units')
    plt.xlabel('Look-Back Period')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    # Plot MSE by dropout_rate
    plt.subplot(2, 2, 2)
    dropout_results = results_df.groupby('dropout_rate')['MSE'].mean().reset_index()
    plt.bar(dropout_results['dropout_rate'].astype(str), dropout_results['MSE'])
    plt.title('MSE by Dropout Rate')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)

    # Plot MSE by batch_size
    plt.subplot(2, 2, 3)
    batch_results = results_df.groupby('batch_size')['MSE'].mean().reset_index()
    plt.bar(batch_results['batch_size'].astype(str), batch_results['MSE'])
    plt.title('MSE by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)

    # Plot training time by combination
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['MSE'], results_df['training_time'])
    plt.title('Training Time vs MSE')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Training Time (s)')
    plt.grid(True)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(results_dir, f'{symbol}_hyperparameter_tuning.png')
    plt.savefig(plot_file)
    print(f"Hyperparameter tuning plot saved to {plot_file}")

    return results_df

if __name__ == "__main__":
    # Run hyperparameter tuning
    tune_hyperparameters(
        symbol='AAPL',
        look_backs=[30, 60, 90],
        lstm_units=[50, 100],
        dropout_rates=[0.2, 0.3],
        batch_sizes=[16, 32, 64],
        epochs=50
    )
