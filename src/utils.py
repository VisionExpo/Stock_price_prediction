import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class ModelTracker:
    def __init__(self):
        self.metrics_dir = "model_metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def save_metrics(self, metrics, model_params, stock_symbol):
        """Save model metrics and parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_symbol}_metrics_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "stock_symbol": stock_symbol,
            "metrics": metrics,
            "parameters": model_params
        }
        
        with open(os.path.join(self.metrics_dir, filename), "w") as f:
            json.dump(data, f, indent=4)
            
    def load_latest_metrics(self, stock_symbol):
        """Load the most recent metrics for a stock symbol"""
        files = [f for f in os.listdir(self.metrics_dir) 
                if f.startswith(f"{stock_symbol}_metrics_")]
        if not files:
            return None
            
        latest_file = max(files)
        with open(os.path.join(self.metrics_dir, latest_file), "r") as f:
            return json.load(f)
            
    def get_all_metrics(self, stock_symbol):
        """Get all historical metrics for a stock symbol"""
        metrics_list = []
        for file in os.listdir(self.metrics_dir):
            if file.startswith(f"{stock_symbol}_metrics_"):
                with open(os.path.join(self.metrics_dir, file), "r") as f:
                    metrics_list.append(json.load(f))
        return metrics_list

def plot_predictions(y_true, y_pred, title="Stock Price Prediction"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    return plt.gcf()

def plot_metrics_history(metrics_list):
    """Plot historical metrics over time"""
    if not metrics_list:
        return None
        
    dates = [m['timestamp'] for m in metrics_list]
    mse = [m['metrics']['MSE'] for m in metrics_list]
    rmse = [m['metrics']['RMSE'] for m in metrics_list]
    mae = [m['metrics']['MAE'] for m in metrics_list]
    r2 = [m['metrics']['R2 Score'] for m in metrics_list]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
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
    return fig