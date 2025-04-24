# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-04-24

### Added
- Initial release of the Stock Price Prediction application
- LSTM-based model for predicting stock prices
- Streamlit web interface for interactive model training and prediction
- Real-time stock data fetching using Tiingo API
- Comprehensive model evaluation metrics
- Future price prediction with visualization
- Deployment configuration for Render

### Changed
- Switched from yfinance to Tiingo API for more reliable data access

### Fixed
- Resolved WebSocketHandler issues with Streamlit
- Fixed model training and prediction pipeline
- Corrected data preprocessing for better accuracy
