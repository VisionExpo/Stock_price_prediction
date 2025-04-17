import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_retrieval import retrieve_data
from src.data_preprocessing import preprocess_data, create_dataset
from src.model import create_model, train_model
from src.predict import predict_future
from src.utils import ModelTracker, plot_metrics_history

# Page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Project information
st.title("Stock Price Prediction using LSTM")

with st.expander("About this Project", expanded=True):
    st.markdown("""
    ### Stock Price Prediction using LSTM Neural Networks
    
    This application uses Long Short-Term Memory (LSTM) neural networks to predict future stock prices. 
    LSTMs are particularly well-suited for this task because they can:
    - Remember long-term patterns in time series data
    - Handle the non-linear nature of stock markets
    - Process sequential data effectively
    
    #### Features:
    - 🔍 Historical data visualization
    - 📈 Interactive model training with customizable parameters
    - 🤖 Advanced LSTM architecture with multiple layers
    - 📊 Comprehensive performance metrics
    - 🎯 Future price predictions
    - 📉 Model performance tracking over time
    
    #### How it works:
    1. Historical stock data is retrieved dynamically
    2. Data is preprocessed and normalized
    3. LSTM model is trained on the processed data
    4. Model predicts future stock prices
    5. Performance metrics are tracked and visualized
    
    > Note: Stock price prediction is inherently uncertain. This tool should be used for educational purposes only.
    """)

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
days_of_history = st.sidebar.slider("Days of Historical Data", 100, 1000, 365)
action = st.sidebar.radio("Choose Action", ["View Historical Data", "Train Model", "Make Predictions", "View Model Performance"])

# Initialize model tracker
model_tracker = ModelTracker()

# Main content
if ticker:
    try:
        if action == "View Historical Data":
            st.subheader(f"Historical Data for {ticker}")
            with st.spinner("Fetching data..."):
                df = retrieve_data(ticker, days=days_of_history)
                
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.index, df.values)
            plt.title(f"{ticker} Stock Price History")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.dataframe(df.tail())
            
            # Show basic statistics with explanations
            st.subheader("Statistical Analysis")
            stats = df.describe()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${df[-1]:.2f}")
                st.metric("Average Price", f"${stats['mean']:.2f}")
                st.metric("Standard Deviation", f"${stats['std']:.2f}")
            
            with col2:
                st.metric("Highest Price", f"${stats['max']:.2f}")
                st.metric("Lowest Price", f"${stats['min']:.2f}")
                price_range = stats['max'] - stats['min']
                st.metric("Price Range", f"${price_range:.2f}")

        elif action == "Train Model":
            st.subheader("Train New Model")
            
            # Model parameters in tabs
            tab1, tab2, tab3 = st.tabs(["Basic Parameters", "Advanced Parameters", "Training Settings"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    epochs = st.slider("Number of Epochs", 10, 200, 100)
                    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
                with col2:
                    look_back = st.slider("Look Back Period (days)", 10, 100, 60)
                    train_split = st.slider("Training Data Split", 0.5, 0.9, 0.8)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    lstm_units = st.slider("LSTM Units", 32, 256, 100)
                    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
                with col2:
                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.001, 0.01, 0.1],
                        value=0.001
                    )
            
            with tab3:
                early_stopping = st.checkbox("Use Early Stopping", value=True)
                if early_stopping:
                    patience = st.slider("Early Stopping Patience", 5, 50, 20)
                
                validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Collect all parameters
                    model_params = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "look_back": look_back,
                        "train_split": train_split,
                        "lstm_units": lstm_units,
                        "dropout_rate": dropout_rate,
                        "learning_rate": learning_rate,
                        "validation_split": validation_split,
                        "early_stopping": early_stopping,
                        "patience": patience if early_stopping else None
                    }
                    
                    # Get and preprocess data
                    df = retrieve_data(ticker, days=days_of_history)
                    train_data, test_data, scaler = preprocess_data(df)
                    
                    # Create sequences
                    x_train, y_train = create_dataset(train_data, time_step=look_back)
                    x_test, y_test = create_dataset(test_data, time_step=look_back)
                    
                    # Reshape input for LSTM
                    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                    
                    # Create and train model
                    model = create_model(
                        (look_back, 1),
                        lstm_units=lstm_units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # Training progress bar
                    progress_text = "Training in progress. Please wait..."
                    progress_bar = st.progress(0)
                    
                    # Custom callback for progress bar
                    class ProgressCallback:
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                    
                    callbacks = [ProgressCallback()]
                    if early_stopping:
                        from tensorflow.keras.callbacks import EarlyStopping
                        callbacks.append(EarlyStopping(
                            monitor='val_loss',
                            patience=patience,
                            restore_best_weights=True
                        ))
                    
                    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=callbacks
                    )
                    
                    # Save model
                    model.save("stock_model.h5")
                    st.success("Model trained successfully!")
                    
                    # Calculate and display metrics
                    train_pred = model.predict(x_train, verbose=0)
                    test_pred = model.predict(x_test, verbose=0)
                    
                    # Inverse transform predictions
                    train_pred = scaler.inverse_transform(train_pred)
                    test_pred = scaler.inverse_transform(test_pred)
                    y_train_inv = scaler.inverse_transform([y_train])
                    y_test_inv = scaler.inverse_transform([y_test])
                    
                    # Calculate metrics
                    train_metrics = {
                        'MSE': mean_squared_error(y_train_inv.T, train_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_train_inv.T, train_pred)),
                        'MAE': mean_absolute_error(y_train_inv.T, train_pred),
                        'R2 Score': r2_score(y_train_inv.T, train_pred)
                    }
                    
                    test_metrics = {
                        'MSE': mean_squared_error(y_test_inv.T, test_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test_inv.T, test_pred)),
                        'MAE': mean_absolute_error(y_test_inv.T, test_pred),
                        'R2 Score': r2_score(y_test_inv.T, test_pred)
                    }
                    
                    # Save metrics
                    model_tracker.save_metrics(test_metrics, model_params, ticker)
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Training History", "Performance Metrics", "Predictions vs Actual"])
                    
                    with tab1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(history.history['loss'], label='Training Loss')
                        ax.plot(history.history['val_loss'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.set_title('Training History')
                        st.pyplot(fig)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Training Metrics")
                            for metric, value in train_metrics.items():
                                st.metric(metric, f"{value:.4f}")
                            
                        with col2:
                            st.subheader("Testing Metrics")
                            for metric, value in test_metrics.items():
                                st.metric(metric, f"{value:.4f}")
                    
                    with tab3:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.scatter(y_test_inv.T, test_pred, alpha=0.5)
                        ax.plot([y_test_inv.min(), y_test_inv.max()], 
                               [y_test_inv.min(), y_test_inv.max()], 
                               'r--', lw=2)
                        ax.set_xlabel('Actual Price')
                        ax.set_ylabel('Predicted Price')
                        ax.set_title('Prediction vs Actual (Test Set)')
                        st.pyplot(fig)

        elif action == "Make Predictions":
            st.subheader("Future Price Predictions")
            
            try:
                model = load_model("stock_model.h5")
                
                # Get the most recent data
                df = retrieve_data(ticker, days=days_of_history)
                _, _, scaler = preprocess_data(df)
                
                # Get the sequence length the model expects
                sequence_length = model.input_shape[1]
                
                # Prepare last sequence_length days for prediction
                last_sequence = df[-sequence_length:].values
                last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
                
                # Make prediction
                n_days = st.slider("Number of days to predict", 5, 60, 30)
                
                # Add confidence interval option
                show_confidence = st.checkbox("Show Confidence Interval", value=True)
                if show_confidence:
                    n_simulations = st.slider("Number of Monte Carlo simulations", 10, 100, 50)
                    confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
                
                with st.spinner("Generating predictions..."):
                    predictions = predict_future(model, last_sequence_scaled, n_steps=n_days)
                    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                    
                    # Create dates for predictions
                    last_date = pd.to_datetime(df.index[-1])
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
                    
                    # Create DataFrame with predictions
                    pred_df = pd.DataFrame(predictions, index=future_dates, columns=['Predicted Price'])
                    
                    if show_confidence:
                        # Generate confidence intervals using Monte Carlo simulation
                        simulations = []
                        for _ in range(n_simulations):
                            sim_pred = predict_future(model, last_sequence_scaled, n_steps=n_days)
                            sim_pred = scaler.inverse_transform(sim_pred.reshape(-1, 1)).flatten()
                            simulations.append(sim_pred)
                        
                        simulations = np.array(simulations)
                        lower_percentile = (100 - confidence_level) / 2
                        upper_percentile = 100 - lower_percentile
                        
                        lower_bound = np.percentile(simulations, lower_percentile, axis=0)
                        upper_bound = np.percentile(simulations, upper_percentile, axis=0)
                        
                        pred_df['Lower Bound'] = lower_bound
                        pred_df['Upper Bound'] = upper_bound
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df.index[-60:], df[-60:].values, label='Historical Data')
                    ax.plot(future_dates, predictions, label='Predictions', linestyle='--')
                    
                    if show_confidence:
                        ax.fill_between(future_dates, 
                                      lower_bound, upper_bound, 
                                      alpha=0.2, 
                                      label=f'{confidence_level}% Confidence Interval')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Display prediction table with metrics
                    st.subheader("Predicted Prices")
                    st.dataframe(pred_df)
                    
                    # Add price change analysis
                    current_price = df[-1]
                    final_pred_price = pred_df['Predicted Price'][-1]
                    price_change = final_pred_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Predicted Final Price", f"${final_pred_price:.2f}")
                    with col3:
                        st.metric("Expected Change", 
                                f"${price_change:.2f}", 
                                f"{price_change_pct:+.2f}%")
                    
                    # Download predictions
                    csv = pred_df.to_csv()
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f'{ticker}_predictions.csv',
                        mime='text/csv'
                    )
                
            except FileNotFoundError:
                st.error("No trained model found. Please train a model first!")
        
        elif action == "View Model Performance":
            st.subheader("Model Performance History")
            
            metrics_list = model_tracker.get_all_metrics(ticker)
            
            if not metrics_list:
                st.warning("No historical metrics found for this stock symbol. Train a model first.")
            else:
                # Display metrics history
                fig = plot_metrics_history(metrics_list)
                if fig:
                    st.pyplot(fig)
                
                # Show detailed metrics table
                st.subheader("Historical Model Metrics")
                metrics_df = pd.DataFrame([
                    {
                        'Timestamp': m['timestamp'],
                        'MSE': m['metrics']['MSE'],
                        'RMSE': m['metrics']['RMSE'],
                        'MAE': m['metrics']['MAE'],
                        'R² Score': m['metrics']['R2 Score'],
                        'LSTM Units': m['parameters']['lstm_units'],
                        'Epochs': m['parameters']['epochs'],
                        'Batch Size': m['parameters']['batch_size'],
                        'Learning Rate': m['parameters']['learning_rate']
                    }
                    for m in metrics_list
                ])
                
                st.dataframe(metrics_df.sort_values('Timestamp', ascending=False))
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")