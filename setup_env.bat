@echo off
echo Setting up environment for Stock Price Prediction...

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    echo # Tiingo API Key > .env
    echo TIINGO_API_KEY=your_tiingo_api_key_here >> .env
    echo.
    echo Please edit the .env file and add your Tiingo API key.
) else (
    echo .env file already exists.
)

echo.
echo Setup complete! You can now run the application with:
echo streamlit run app.py
echo.
