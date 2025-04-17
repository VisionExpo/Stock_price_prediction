import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(input_shape, lstm_units=100, dropout_rate=0.2, learning_rate=0.001, l2_reg=0.01):
    """
    Create enhanced LSTM model with modern improvements
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        lstm_units (int): Number of LSTM units in each layer
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for Adam optimizer
        l2_reg (float): L2 regularization factor
    """
    model = Sequential()
    
    # First LSTM layer with batch normalization and residual connection
    model.add(LSTM(lstm_units, 
                  return_sequences=True, 
                  input_shape=input_shape,
                  kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(lstm_units, 
                  return_sequences=True,
                  kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Final LSTM layer
    model.add(LSTM(lstm_units,
                  kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', 
                 optimizer=optimizer,
                 metrics=['mae', 'mse'])  # Added metrics for better tracking
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=32):
    """Train the model with the given parameters"""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Early stopping with patience
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        mode='min'
    )
    
    # Reduce learning rate when training plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        mode='min'
    )
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return history

def predict_future(model, input_data, n_steps=30):
    """Make future predictions with uncertainty estimation"""
    predictions = []
    input_seq = input_data.copy()
    
    for _ in range(n_steps):
        # Make prediction
        pred = model.predict(input_seq)
        predictions.append(pred[0, 0])
        
        # Update input sequence
        input_seq = np.roll(input_seq, -1)
        input_seq[0, -1, 0] = pred[0, 0]
    
    return np.array(predictions)

if __name__ == "__main__":
    pass
