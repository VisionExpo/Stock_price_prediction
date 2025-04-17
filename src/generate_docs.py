import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import seaborn as sns
from data_retrieval import retrieve_data
from data_preprocessing import preprocess_data, create_dataset
from model import create_model
from utils import ModelTracker

def generate_architecture_diagram(model, path):
    """Generate and save model architecture diagram"""
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)

def generate_metrics_visualization(path):
    """Generate and save metrics history visualization"""
    tracker = ModelTracker()
    metrics_list = tracker.get_all_metrics('AAPL')  # Using AAPL as example
    if metrics_list:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = [m['timestamp'] for m in metrics_list]
        mse = [m['metrics']['MSE'] for m in metrics_list]
        rmse = [m['metrics']['RMSE'] for m in metrics_list]
        mae = [m['metrics']['MAE'] for m in metrics_list]
        r2 = [m['metrics']['R2 Score'] for m in metrics_list]
        
        ax1.plot(dates, mse)
        ax1.set_title('MSE History')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.plot(dates, rmse)
        ax2.set_title('RMSE History')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.plot(dates, mae)
        ax3.set_title('MAE History')
        ax3.tick_params(axis='x', rotation=45)
        
        ax4.plot(dates, r2)
        ax4.set_title('R² Score History')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def generate_prediction_visualization(model, data, scaler, path):
    """Generate and save prediction visualization"""
    sequence_length = model.input_shape[1]
    x_test, y_test = create_dataset(data, time_step=sequence_length)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    predictions = model.predict(x_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform([y_test]).T
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Stock Price Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(path)
    plt.close()

def generate_training_history(model, x_train, y_train, x_test, y_test, path):
    """Generate and save training history visualization"""
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,  # Using fewer epochs for demonstration
        batch_size=32,
        verbose=0
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

def generate_confidence_intervals(model, data, scaler, path):
    """Generate and save confidence intervals visualization"""
    sequence_length = model.input_shape[1]
    last_sequence = data[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    
    # Generate multiple predictions with Monte Carlo
    n_simulations = 100
    n_days = 30
    simulations = []
    
    x_input = last_sequence_scaled.reshape((1, sequence_length, 1))
    for _ in range(n_simulations):
        pred = model.predict(x_input, verbose=0)
        simulations.append(pred[0, 0])
    
    simulations = np.array(simulations)
    mean_pred = np.mean(simulations)
    std_pred = np.std(simulations)
    
    plt.figure(figsize=(12, 6))
    # Plot historical data
    historical = scaler.inverse_transform(data[-60:].values.reshape(-1, 1))
    plt.plot(range(60), historical, label='Historical')
    
    # Plot prediction with confidence interval
    pred_point = scaler.inverse_transform([[mean_pred]])[0, 0]
    lower_bound = scaler.inverse_transform([[mean_pred - 2*std_pred]])[0, 0]
    upper_bound = scaler.inverse_transform([[mean_pred + 2*std_pred]])[0, 0]
    
    plt.plot([59, 60], [historical[-1][0], pred_point], 'r--', label='Mean Prediction')
    plt.fill_between([59, 60], 
                    [historical[-1][0], lower_bound],
                    [historical[-1][0], upper_bound],
                    color='gray', alpha=0.2, 
                    label='95% Confidence Interval')
    
    plt.title('Stock Price Prediction with Confidence Intervals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(path)
    plt.close()

def generate_demo_screenshot(path):
    """Generate and save demo screenshot visualization"""
    # Create a mock dashboard layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mock data
    x = np.linspace(0, 100, 100)
    y1 = np.sin(x/10) * 100 + 200
    y2 = y1 + np.random.normal(0, 5, 100)
    
    # Plot 1: Stock Price History
    ax1.plot(x, y1)
    ax1.set_title('Stock Price History')
    
    # Plot 2: Prediction
    ax2.plot(x, y1, label='Actual')
    ax2.plot(x, y2, label='Predicted')
    ax2.legend()
    ax2.set_title('Price Prediction')
    
    # Plot 3: Training Metrics
    ax3.plot(x[:50], np.exp(-x[:50]/30))
    ax3.set_title('Training Loss')
    
    # Plot 4: Performance Metrics
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    values = [0.02, 0.14, 0.12, 0.95]
    ax4.bar(metrics, values)
    ax4.set_title('Model Performance')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    # Create paths
    docs_path = os.path.join('docs', 'images')
    paths = {
        'architecture': os.path.join(docs_path, 'architecture.png'),
        'metrics': os.path.join(docs_path, 'metrics.png'),
        'prediction': os.path.join(docs_path, 'prediction.png'),
        'training': os.path.join(docs_path, 'training.png'),
        'confidence': os.path.join(docs_path, 'confidence.png'),
        'demo': os.path.join(docs_path, 'demo.png')
    }
    
    # Get sample data
    df = retrieve_data('AAPL', days=365)
    train_data, test_data, scaler = preprocess_data(df)
    
    # Create sequences
    sequence_length = 60
    x_train, y_train = create_dataset(train_data, time_step=sequence_length)
    x_test, y_test = create_dataset(test_data, time_step=sequence_length)
    
    # Reshape for LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    # Create model
    model = create_model((sequence_length, 1))
    
    # Generate visualizations
    print("Generating architecture diagram...")
    generate_architecture_diagram(model, paths['architecture'])
    
    print("Generating metrics visualization...")
    generate_metrics_visualization(paths['metrics'])
    
    print("Generating prediction visualization...")
    generate_prediction_visualization(model, test_data, scaler, paths['prediction'])
    
    print("Generating training history visualization...")
    generate_training_history(model, x_train, y_train, x_test, y_test, paths['training'])
    
    print("Generating confidence intervals visualization...")
    generate_confidence_intervals(model, df, scaler, paths['confidence'])
    
    print("Generating demo screenshot...")
    generate_demo_screenshot(paths['demo'])
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main()