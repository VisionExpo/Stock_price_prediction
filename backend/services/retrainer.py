import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def trigger_retraining():
    """
    Triggers the final model training script as a background process.
    Returns the process object.
    """
    script_path = "pipelines/train_final_model.py"
    logging.info(f"Triggering model retraining script: {script_path}")
    
    try:
        # Using Popen to run the script in the background without blocking the API
        process = subprocess.Popen([sys.executable, script_path])
        logging.info(f"Retraining process started with PID: {process.pid}")
        return process
    except Exception as e:
        logging.error(f"Failed to start retraining script: {e}")
        return None

if __name__ == '__main__':
    # Example of how to run this service
    print("Starting a retraining job...")
    retraining_process = trigger_retraining()
    if retraining_process:
        retraining_process.wait() # Wait for the process to complete for this example
        print("Retraining job finished.")