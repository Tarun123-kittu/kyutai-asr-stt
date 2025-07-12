# Kyutai Real-Time Streaming STT Service
# Uses moshi library with Python 3.12 and CUDA
FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Using a single RUN command for better Docker layer caching
RUN pip install --no-cache-dir --upgrade pip

# 1. Install PyTorch with CUDA support (kept separate for caching)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install API, Audio, and Moshi Streaming dependencies
RUN pip install --no-cache-dir \
    # API
    fastapi \
    uvicorn[standard] \
    websockets \
    loguru \
    python-multipart \
    # Audio
    numpy \
    librosa \
    soundfile \
    # Moshi Streaming
    "sphn<0.2" \
    "moshi==0.2.8" \
    "sentencepiece"

# Create directories for logs (model cache is handled by Hugging Face default)
RUN mkdir -p /app/logs

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Add healthcheck to ensure the model has loaded before marking as healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]