import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Define the LSTM Model Architecture ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        
        # Fully connected linear layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        
        # We only need the output of the LSTM layer, not the hidden states
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        
        # Pass the output of the last time step to the linear layer
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- 2. Define the Training Function ---
def train_lstm_model(data_dir: Path, model_output_path: Path, epochs: int, learning_rate: float):
    """
    Loads sequence data, trains the LSTM model, and saves the trained model.
    """
    # --- Setup Device (GPU/CPU) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    logging.info("Loading sequence data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    # --- Convert to PyTorch Tensors ---
    # Reshape y_train to be [samples, 1]
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1).to(device)

    # --- Initialize Model, Loss, and Optimizer ---
    # The input_size is the number of features in our sequence data
    input_size = X_train.shape[2] 
    
    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logging.info(f"Model initialized:\n{model}")

    # --- Training Loop ---
    logging.info("Starting model training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    logging.info("Training complete.")

    # --- Save Model ---
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    logging.info(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # --- Configuration ---
    SEQUENCE_DATA_DIR = Path("data/sequences_sentiment")
    MODEL_PATH = Path("models/lstm_v2_sentiment.pt")
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # --- Execution ---
    train_lstm_model(
        data_dir=SEQUENCE_DATA_DIR,
        model_output_path=MODEL_PATH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )