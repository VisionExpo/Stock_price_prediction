import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("Stock Market Scam Dashborad 📈")

st.write("Connecting to the backend API...")

# The URL points to the backend service name from docker-compose
API_URL = "http://backend:8000"

try:
    response = requests.get(API_URL)
    if response.status_code == 200:
        st.success(f"Successfully connected to the backend! ✨")
        st.json(response.json())
    else:
        st.error(f"Failed to connect. Status code : {response.status_code}")
except requests.exceptions.ConnectionError as e:
    st.error(f"Connectin Error: Could not connect to the backend at {API_URL}.")
    st.info("Please ensure the backend service is running.")
