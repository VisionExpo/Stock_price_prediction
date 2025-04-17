import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from data_retrieval import retrieve_data
from data_preprocessing import preprocess_data, create_dataset
from model import create_model, train_model
from predict import predict_future, save_model

def main():
    choice = input("Do you want to train the model or make predictions? (train/predict): ").strip().lower()
    
    if choice == 'train':
        ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
        
        # Get and preprocess data
        df = retrieve_data(ticker)
        train_data, test_data, scaler = preprocess_data(df)
        
        # Create sequences for LSTM
        x_train, y_train = create_dataset(train_data, time_step=60)
        x_test, y_test = create_dataset(test_data, time_step=60)
        
        # Reshape input for LSTM [samples, time steps, features]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        # Create and train model
        model = create_model((x_train.shape[1], 1))
        model = train_model(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=32)
        
        # Save the model
        save_model(model, "stock_model.h5")
        print(f"Model trained and saved as 'stock_model.h5' for {ticker}")
    
    elif choice == 'predict':
        model = load_model("stock_model.h5")
        ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
        df = retrieve_data(ticker)
        _, _, scaler = preprocess_data(df)
        
        # Get the sequence length the model expects
        sequence_length = model.input_shape[1]
        
        # Prepare last sequence_length days for prediction
        last_sequence = df[-sequence_length:].values
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        # Make prediction
        predictions = predict_future(model, last_sequence_scaled, n_steps=30)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        
        print(f"\nPredicted stock prices for {ticker} over next 30 days:")
        for i, price in enumerate(predictions, 1):
            print(f"Day {i}: ${price[0]:.2f}")

if __name__ == "__main__":
    main()
