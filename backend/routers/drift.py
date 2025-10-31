from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse, HTMLResponse
import subprocess
import sys
from pathlib import Path
import logging

router = APIRouter()

@router.post("/run/drift-detection")
def run_drift_detection():
    script_path = "pipelines/drift_detection_pipeline.py"
    logging.info(f"Triggering drift detection script: {script_path}")
    try:
        subprocess.Popen([sys.executable, script_path])
        return {"message": "Drift detection process started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start drift detection script: {e}")

@router.get("/reports/drift", response_class=HTMLResponse)
def get_drift_report():
    report_path = Path("reports/data_drift_report.html")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Drift report not found.")
    return FileResponse(report_path)