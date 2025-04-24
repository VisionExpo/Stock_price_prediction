#!/bin/bash

echo "Setting up environment for Stock Price Prediction..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "# Environment variables" > .env
    echo "# No API keys required for this project" >> .env
    echo ""
else
    echo ".env file already exists."
fi

echo ""
echo "Setup complete! You can now run the application with:"
echo "streamlit run app.py"
echo ""
