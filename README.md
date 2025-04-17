# Stock Price Prediction Using LSTM

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. The model is designed to predict future stock prices based on historical data, featuring both a command-line interface and an interactive web application built with Streamlit.

![Stock Price Prediction Demo](docs/images/demo.png)

## Features

- 🔍 **Real-time Data**: Fetches real-time stock data using yfinance
- 📈 **Interactive Training**: Fine-tune model parameters through an intuitive interface
- 🤖 **Advanced LSTM Architecture**: Multi-layer LSTM with dropout for robust predictions
- 📊 **Comprehensive Metrics**: Track MSE, RMSE, MAE, and R² scores
- 🎯 **Future Predictions**: Generate price predictions with confidence intervals
- 📉 **Performance Tracking**: Monitor model performance over time
- 📱 **Responsive UI**: User-friendly interface built with Streamlit

## Project Structure

```
├── src/                # Source code directory
│   ├── __init__.py     # Makes src a Python package
│   ├── config.py       # Configuration parameters
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── model.py        # LSTM model definition
│   ├── utils.py        # Utility functions
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   ├── predict.py      # Prediction script
│   └── main.py         # Main pipeline script
├── model_metrics/      # Saved model metrics and history
├── models/             # Saved model files
├── docs/              # Documentation and images
├── app.py             # Streamlit web application
└── requirements.txt   # Required packages
```

## Model Architecture

The LSTM model architecture consists of:
- 3 LSTM layers with customizable units (default: 100)
- Dropout layers for regularization (customizable rate)
- Dense output layer for price prediction
- Adam optimizer with configurable learning rate

![Model Architecture](docs/images/architecture.png)

## Performance Metrics

The model tracks several key performance metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)

Example metrics visualization:
![Metrics History](docs/images/metrics.png)

## Prediction Visualization

The model provides various visualizations:
1. Historical vs Predicted Prices
   ![Price Prediction](docs/images/prediction.png)

2. Training History
   ![Training History](docs/images/training.png)

3. Prediction Confidence Intervals
   ![Confidence Intervals](docs/images/confidence.png)

## Requirements

- Python 3.6+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn
- streamlit
- yfinance

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch the web application:
```bash
streamlit run app.py
```

## Usage

### Web Interface

1. **View Historical Data**
   - Enter a stock symbol
   - Adjust the time range
   - View price history and statistics

2. **Train Model**
   - Configure basic parameters:
     - Epochs
     - Batch size
     - Look-back period
   - Set advanced parameters:
     - LSTM units
     - Dropout rate
     - Learning rate
   - Enable/disable early stopping

3. **Make Predictions**
   - Select prediction timeframe
   - Enable confidence intervals
   - View predicted prices and trends
   - Download prediction results

4. **Track Performance**
   - View historical metrics
   - Compare model versions
   - Analyze performance trends

### Configuration

Customize model parameters in `src/config.py`:

```python
STOCK_SYMBOL = 'AAPL'        # Default stock symbol
TRAIN_TEST_SPLIT = 0.8      # Training data ratio
LSTM_UNITS = 100           # LSTM layer units
DROPOUT_RATE = 0.2        # Dropout rate
EPOCHS = 100             # Training epochs
BATCH_SIZE = 32         # Batch size
LOOK_BACK = 60         # Time steps to look back
LEARNING_RATE = 0.001 # Adam optimizer learning rate
```

## Model Performance

Example performance metrics on various stocks:

| Stock | MSE    | RMSE   | MAE    | R² Score |
|-------|---------|--------|--------|----------|
| AAPL  | 12.45   | 3.52   | 2.89   | 0.95     |
| GOOGL | 15.67   | 3.96   | 3.12   | 0.93     |
| MSFT  | 10.23   | 3.19   | 2.54   | 0.96     |

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit team for the amazing web app framework
- yfinance for providing stock data access
