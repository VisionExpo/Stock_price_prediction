import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import math
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the model definition from the previous script
from pipelines.train_transformer_pipeline import TransformerModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SEQUENCE_DATA_DIR = Path("data/sequences_sentiment")
MODEL_PATH = Path("models/transformer_champion.pt")
EPOCHS = 100 # We'll use a fixed number of epochs

# --- BEST HYPERPARAMETERS (from your Optuna run) ---
BEST_PARAMS = {
    'learning_rate': 0.00012043894474774808,
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dropout': 0.1057777547471907
}

def train_final_model():
    """
    Trains the final champion model using the best hyperparameters on the full dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info("Loading FULL training dataset...")
    # Load the entire training dataset (no validation split)
    X_train = np.load(SEQUENCE_DATA_DIR / "X_train.npy")
    y_train = np.load(SEQUENCE_DATA_DIR / "y_train.npy")

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1).to(device)

    # Initialize model with the best parameters
    input_size = X_train.shape[2]
    model = TransformerModel(
        input_size=input_size,
        d_model=BEST_PARAMS['d_model'],
        nhead=BEST_PARAMS['nhead'],
        num_encoder_layers=BEST_PARAMS['num_encoder_layers'],
        dropout=BEST_PARAMS['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_PARAMS['learning_rate'])
    
    logging.info("Starting final model training with optimal hyperparameters...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')
            
    logging.info("Final training complete.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Champion model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_final_model()