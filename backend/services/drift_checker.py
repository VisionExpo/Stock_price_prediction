import pandas as pd
from pathlib import Path
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_for_drift(data_path: Path, reference_end_date: str, drift_report_path: Path) -> bool:
    """
    Checks for data drift and returns True if drift is detected.
    Also saves the full HTML report.
    """
    logging.info("Checking for data drift...")
    
    try:
        df = pd.read_csv(data_path, parse_dates=['Date'])
        
        reference_df = df[df['Date'] < reference_end_date]
        current_df = df[df['Date'] >= reference_end_date]

        if reference_df.empty or current_df.empty:
            logging.warning("Cannot check drift: data splits are empty.")
            return False

        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_df, current_data=current_df)
        
        # Save the full report for inspection
        drift_report_path.parent.mkdir(parents=True, exist_ok=True)
        drift_report.save(filename=str(drift_report_path))
        logging.info(f"Full drift report saved to {drift_report_path}")

        # --- The core logic: check the report's JSON output for drift ---
        report_dict = drift_report.as_dict()
        
        # We access the summary provided by the DataDriftPreset
        drift_detected = False
        if 'metrics' in report_dict and len(report_dict['metrics']) > 0:
            drift_detected = report_dict['metrics'][0]['result'].get('dataset_drift', False)
        else:
            logging.warning("Could not parse drift report structure.")
        
        if drift_detected:
            logging.warning("Data drift DETECTED!")
        else:
            logging.info("No significant data drift detected.")
            
        return drift_detected

    except Exception as e:
        logging.error(f"An error occurred during drift check: {e}")
        return False

if __name__ == '__main__':
    # Example of how to run this service
    FUSED_DATA_PATH = Path("data/processed/final_fused_data.csv")
    DRIFT_REPORT_PATH = Path("reports/data_drift_report.html")
    SPLIT_DATE = "2023-01-01"
    
    drift_found = check_for_drift(
        data_path=FUSED_DATA_PATH,
        reference_end_date=SPLIT_DATE,
        drift_report_path=DRIFT_REPORT_PATH
    )
    print(f"Drift detected: {drift_found}")