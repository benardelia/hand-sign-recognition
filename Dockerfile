# Use official Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001

# Set working directory
WORKDIR /app

# libgomp1: OpenMP runtime for scikit-learn / TensorFlow. No OpenCV/MediaPipe system libs are
# needed: landmarks are extracted in the browser / phone app, the server only handles numbers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Ensure Data and Model directories exist
RUN mkdir -p /app/Data /app/Model

# Expose the application port
EXPOSE 5001

# Default command for production using Gunicorn WSGI server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120"]
