# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for video processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entrypoint to run the app with Gunicorn
# Gunicorn will listen on the port specified by the PORT env var (provided by App Engine)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
