import streamlit as st
import requests
import time

st.set_page_config(page_title="Drift Analysis", page_icon="📊", layout="wide")
st.title("📊 Data Drift Dashboard")

BASE_API_URL = "http://backend:8000"

st.info("This page compares the training data with recent data to detect 'data drift'. If features drift significantly, the model may need to be retrained.")

if st.button("Generate / Refresh Drift Report"):
    with st.spinner("Running drift detection pipeline... This may take a minute."):
        try:
            run_response = requests.post(f"{BASE_API_URL}/run/drift-detection")
            run_response.raise_for_status()
            st.toast("Drift analysis started...")

            # Poll for report availability instead of fixed sleep
            max_attempts = 60  # 60 seconds max
            for attempt in range(max_attempts):
                time.sleep(5)
                report_response = requests.get(f"{BASE_API_URL}/reports/drift")
                if report_response.status_code == 200:
                    st.toast("Report generated successfully! Displaying below.")
                    st.components.v1.html(report_response.text, height=1000, scrolling=True)
                    break
                elif attempt == max_attempts - 1:
                    st.error("Report generation timed out. Please try again.")
                else:
                    st.info(f"Waiting for report... ({attempt + 1}/{max_attempts})")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to communicate with the API: {e}")

st.markdown("---")
st.markdown("Click the button above to generate and view the latest drift report.")