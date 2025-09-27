import pandas as pd
from pathlib import Path
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def score_headlines(input_path: Path, output_path: Path, model_name: str = "ProsusAI/finbert"):
    """
    Loads raw news data, scores sentiment for each headline using FinBERT,
    and saves the aggregated daily scores.
    """
    # --- Setup Device (GPU/CPU) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        # --- Load Model and Tokenizer ---
        logging.info(f"Loading FinBERT model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        logging.info("Model and tokenizer loaded successfully.")

        # --- Load Data ---
        df = pd.read_csv(input_path, parse_dates=['Date'])
        headlines = df['Headline'].tolist()
        
        if not headlines:
            logging.warning("No headlines to process.")
            return

        # --- Get Sentiment Scores (in batches for efficiency) ---
        logging.info(f"Scoring sentiment for {len(headlines)} headlines...")
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert logits to probabilities
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT labels: 0=positive, 1=negative, 2=neutral
        # We calculate a single score: positive_prob - negative_prob
        positive = predictions[:, 0].tolist()
        negative = predictions[:, 1].tolist()
        
        df['sentiment_score'] = [p - n for p, n in zip(positive, negative)]
        logging.info("Sentiment scoring complete.")

        # --- Aggregate Scores by Day ---
        # A stock can have multiple news articles per day. We'll take the average sentiment.
        daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
        
        # Save the processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_sentiment.to_csv(output_path, index=False)
        
        logging.info(f"Successfully aggregated daily sentiment scores and saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during sentiment scoring: {e}")


if __name__ == "__main__":
    # Define paths
    RAW_NEWS_PATH = Path("data/raw/raw_news_data.csv")
    PROCESSED_SENTIMENT_PATH = Path("data/processed/processed_sentiment_scores.csv")
    
    # Execute the scoring function
    score_headlines(input_path=RAW_NEWS_PATH, output_path=PROCESSED_SENTIMENT_PATH)