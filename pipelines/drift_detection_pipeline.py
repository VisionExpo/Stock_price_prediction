import pandas as pd
from pathlib import Path
import logging
from evidently import Report
from evidently.presets import DataDriftPreset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_drift_report(data_path: Path, output_path: Path, reference_end_date: str):
    """
    Generates a data drift report comparing two time periods of a dataset.
    
    Args:
        data_path (Path): Path to the full, fused dataset.
        output_path (Path): Path to save the HTML report.
        reference_end_date (str): The date to split the data info into reference and current sets.
        """
    
    logging.info("Starting data drift report generation...")

    try:
        # Load the dataset
        df= pd.read_csv(data_path, parse_dates=['Date'])
        logging.info(f"Loaded data with shape: {df.shape}")


        # Split the data into reference(training) and current(production) sets
        reference_df = df[df['Date'] < reference_end_date]
        current_df = df[df['Date'] >= reference_end_date]

        if reference_df.empty or current_df.empty:
            logging.error("Could not create both reference and current dataframes. Check date split.")
            return
        
        logging.info(f"Reference data shape: {reference_df.shape}")
        logging.info(f"Current data shape: {current_df.shape}")

        # Create the drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
        ])

        logging.info("Running Evidently AI analysis...")
        drift_report.run(reference_data=reference_df, current_data=current_df)

        # Save the report as an HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drift_report.save(filename=(str(output_path))

        logging.info(f"Drift report saved successfully to {output_path}")


    except Exception as e:
        logging.error(f"An error occurred during report generation: {e}")


if __name__ == "__main__":
    # Define paths
    FUSED_DATA_PATH = Path("data/processed/final_fused_data.csv")
    DRIFT_REPORT_PATH = Path("reports/data_drift_report.html")

    # We'll use our model's test period start date as the split point
    SPLIT_DATE = "2023-01-01"

    # Execute the function
    generate_drift_report(
        data_path=FUSED_DATA_PATH,
        output_path=DRIFT_REPORT_PATH,
        reference_end_date=SPLIT_DATE
    )