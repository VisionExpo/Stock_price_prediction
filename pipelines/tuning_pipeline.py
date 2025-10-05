import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import math
import optuna
import mlflow
from sklearn.model_selection import train_test_split


# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = 50 # Number of different hyperparameter combinations to test


# --- Model Definitions (Copied from training pipeline) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dropout):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output
    
# --- Data Loading Function ---
def load_data_for_tuning(data_dir: Path):
    logging.info("Loading and splitting data for tuning...")
    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")
    # We split the training data into a new, smaller training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    """ 
    This funtction is called by Optuna for each trial.
    It defines and evaluates a model with a given set of hyperparameters.
    """
    # 1. Define the search space for hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs' : trial.suggest_int('epochs', 10, 100)

    }

    # 2. Start an MLflow run to log this trial
    with mlflow.start_run():
        mlflow.log_params(params)
        
        # Load data
        data_dir = Path("data/sequences_sentiment")
        X_train, X_val, y_train, y_val = load_data_for_tuning(data_dir)

        X_train_tensor = torch.from_numpy(X_train).float().to(DEVICE)
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1).to(DEVICE)
        X_val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
        y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1).to(DEVICE)
        
        # 3. Build and train the model
        input_size = X_train.shape[2]
        model = TransformerModel(
            input_size=input_size,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dropout=params['dropout']
        ).to(DEVICE)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        for epoch in range(params['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # 4. Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
        
        validation_rmse = torch.sqrt(criterion(val_preds, y_val_tensor)).item()
        mlflow.log_metric("validation_rmse", validation_rmse)

    # 5. Return the score to Optuna
    return validation_rmse

# --- Main Execution ---
if __name__ == "__main__":
    # Set the MLflow experiment
    mlflow.set_experiment("Transformer Tuning")
    
    # Create and run the Optuna study
    study = optuna.create_study(direction="minimize") # We want to minimize RMSE
    study.optimize(objective, n_trials=N_TRIALS)

    # Print the best results
    print("\n" + "="*50)
    print("Tuning finished!")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50 + "\n")