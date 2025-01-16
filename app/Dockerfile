# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including Tesseract and Persian language pack
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fas \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Update Pip to the latest version
RUN python -m pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy Tesseract Persian data
COPY tesseract-data/* /usr/share/tesseract-ocr/4.00/tessdata/

# Copy the application code
COPY . .

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV MODEL_PATH=/app/models/model.onnx

# Configure Streamlit to show localhost
ENV STREAMLIT_SERVER_ADDRESS=127.0.0.1

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
