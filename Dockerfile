# Use the official Python 3.11 slim base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files to disk and disable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install basic system packages needed for C dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take advantage of Docker build caching
COPY requirements.txt .

# Upgrade pip and install the dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application package and required folders/files into the container
COPY app/ ./app/
COPY models/ ./models/
COPY datasets/ ./datasets/
COPY run.py .
COPY app.py .

# Expose port 5000, which is the application's binding port
EXPOSE 5000

# Run the Flask application using Gunicorn for production readiness
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "run:app"]
