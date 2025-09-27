# Use updated slim Python base image with security fixes
FROM python:3.11-slim

# Upgrade system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the frontend application code to the container
COPY . .

# Install the necessary Python Packages for Streamlit
RUN pip install streamlit requests plotly

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
