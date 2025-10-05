import streamlit as st

st.set_page_config(
    page_title="AI Stock Predictor - Home",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Welcome to the AI Stock Predictor")

st.markdown("""
This application is an end-to-end MLOps project for stock price prediction.
It uses a containerized architecture with Docker, a FastAPI backend, and this Streamlit frontend.

**Features:**
- **Predictions**: Get next-day price predictions for individual stocks using a Transformer model.
- **Screener**: Run the model on multiple stocks to find potential opportunities.
- **Drift Analysis**: Monitor for data drift between training and recent data.
- **Backtesting**: (Coming Soon) Simulate trading strategies based on model predictions.

**Navigate to a feature using the sidebar on the left.**
""")