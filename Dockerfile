# Use the official NVIDIA CUDA 12.1 base image with cuDNN for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompt and ensure logs appear immediately
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm  -rf /var/lib/apt/lists/*

# Create a 'python' alias for 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layaer caching
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Define the command to run the FastAPI application
CMD [ "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


