# Stock Price Prediction Using LSTM

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. The model is designed to predict future stock prices based on historical data. It includes both a command-line interface and a web application built with Streamlit for easy interaction.

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
│   ├── main.py         # Main pipeline script
│   └── app.py          # Streamlit web application
├── data/               # Directory for stock data
├── models/             # Directory for saved models
├── results/            # Directory for results and plots
├── run.py              # CLI entry point script
├── app.py              # Streamlit app entry point
├── requirements.txt    # Required packages
├── Procfile            # For deployment on Render
├── render.yaml         # Configuration for Render deployment
├── DEPLOYMENT.md       # Deployment instructions
├── .env                # Environment variables (API keys)
└── .gitignore          # Git ignore file
```

## Requirements

- Python 3.6+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn
- python-dotenv
- streamlit
- yfinance
- Pillow

## Setup

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Edit the `.env` file in the project root and add your Tiingo API key:
   ```
   TIINGO_API_KEY=your_api_key_here
   ```

## Usage

### Running the Complete Pipeline

To run the complete pipeline (training and evaluation):

```
python run.py main
```

### Training the Model

To train the model with default parameters:

```
python run.py train
```

To customize the training:

```
python run.py train --symbol AAPL --look-back 60 --epochs 100 --batch-size 32
```

### Evaluating the Model

To evaluate the trained model:

```
python run.py evaluate --plot
```

### Making Predictions

To predict future stock prices:

```
python run.py predict --future-days 30 --plot
```

## Configuration

You can modify the default parameters in `src/config.py`:

- `STOCK_SYMBOL`: Stock symbol to use (default: 'AAPL')
- `TRAIN_TEST_SPLIT`: Train-test split ratio (default: 0.8)
- `LSTM_UNITS`: Number of LSTM units (default: 50)
- `DROPOUT_RATE`: Dropout rate (default: 0.2)
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size (default: 32)
- `LOOK_BACK`: Number of previous time steps to use as input (default: 60)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `VALIDATION_SPLIT`: Validation split ratio (default: 0.1)

## Web Application

The project includes a Streamlit web application that provides a user-friendly interface for training models and making predictions.

To run the web application locally:

```
streamlit run app.py
```

This will start the app on `http://localhost:8501`

### Web App Features

- Train LSTM models with customizable parameters
- Visualize training history and model performance
- Make future stock price predictions
- Interactive charts and data tables
- Download prediction results

## Deployment

The application can be deployed to Render. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## License

This project is licensed under the MIT License.
