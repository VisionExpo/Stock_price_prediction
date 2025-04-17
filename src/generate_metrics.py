from data_retrieval import retrieve_data
from data_preprocessing import preprocess_data, create_dataset
from model import create_model
from utils import ModelTracker
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Get data
    df = retrieve_data('AAPL', days=365)
    train_data, test_data, scaler = preprocess_data(df)
    
    # Create sequences
    sequence_length = 60
    x_train, y_train = create_dataset(train_data, time_step=sequence_length)
    x_test, y_test = create_dataset(test_data, time_step=sequence_length)
    
    # Reshape for LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    tracker = ModelTracker()
    
    # Train multiple models with different parameters to generate metrics history
    lstm_units = [50, 100, 150]
    learning_rates = [0.001, 0.01]
    
    for units in lstm_units:
        for lr in learning_rates:
            print(f"Training model with {units} units and learning rate {lr}")
            model = create_model((sequence_length, 1), lstm_units=units, learning_rate=lr)
            
            # Train model
            model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # Make predictions
            y_pred = model.predict(x_test, verbose=0)
            
            # Inverse transform for metrics calculation
            y_test_inv = scaler.inverse_transform([y_test]).T
            y_pred_inv = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            metrics = {
                'MSE': mean_squared_error(y_test_inv, y_pred_inv),
                'RMSE': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
                'MAE': mean_absolute_error(y_test_inv, y_pred_inv),
                'R2 Score': r2_score(y_test_inv, y_pred_inv)
            }
            
            # Save metrics
            model_params = {
                'lstm_units': units,
                'learning_rate': lr,
                'epochs': 5,
                'batch_size': 32
            }
            
            tracker.save_metrics(metrics, model_params, 'AAPL')
            print("Metrics saved")

if __name__ == "__main__":
    main()