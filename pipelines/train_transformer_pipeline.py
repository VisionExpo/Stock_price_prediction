import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelnamme)s - %(message)s')

# --- 1. Define Positional Encoding ---
# Transformers don't inherently know sequence order, so we add this.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    

# --- 2. Define the Transformer Model Architecture ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # We only need the output from the last element of the sequence
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
            logging.info(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

    logging.info("Training complete.")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    logging.info(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    SEQUENCE_DATA_DIR = Path("data/sequences_sentiment")
    MODEL_PATH = Path("models/transformer_v1.pt")
    EPOCHS = 100
    LEARNING_RATE = 0.0001  # Transformers often require lower learning rates

    train_transformer_model(
        data_dir=SEQUENCE_DATA_DIR,
        model_output_path=MODEL_PATH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )