import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import math

# FIXED: Corrected 'levelnamme' to 'levelname'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Define Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Create a positional encoding matrix of shape (1, max_len, d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 2. Define the Transformer Model Architecture ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        self.input_encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # FIXED: Correctly define the encoder layer first, then the encoder itself
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # We only need the output from the last element of the sequence for prediction
        output = self.decoder(output[:, -1, :])
        return output

# --- 3. Define the Training Function ---
def train_transformer_model(data_dir: Path, model_output_path: Path, epochs: int, learning_rate: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info("Loading sequence data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1).to(device)

    input_size = X_train.shape[2]
    model = TransformerModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logging.info(f"Model initialized:\n{model}")
    logging.info("Starting model training...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # FIXED: Added /epochs for clarity in logging
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    logging.info("Training complete.")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    logging.info(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    SEQUENCE_DATA_DIR = Path("data/sequences_sentiment")
    MODEL_PATH = Path("models/transformer_v1.pt")
    EPOCHS = 100
    LEARNING_RATE = 0.0001

    train_transformer_model(
        data_dir=SEQUENCE_DATA_DIR,
        model_output_path=MODEL_PATH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )