# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the frontend application code to the container
COPY . .

# Install the necessary Python Packages for Streamlit
RUN pip install streamlit requests plotly

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
