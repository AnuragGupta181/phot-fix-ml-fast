# --------------------------------------------------
# Base Image (Python 3.10 - SLIM)
# --------------------------------------------------
FROM python:3.10-slim

# --------------------------------------------------
# Environment settings
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV PIP_NO_CACHE_DIR=1

# --------------------------------------------------
# System dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Set working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Copy project files
# --------------------------------------------------
COPY pyproject.toml ./
COPY src ./src

# --------------------------------------------------
# Upgrade pip
# --------------------------------------------------
RUN pip install --upgrade pip

# --------------------------------------------------
# Install CPU-only PyTorch FIRST
# (Very important: prevents CUDA bloat)
# --------------------------------------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------
# Install remaining dependencies
# --------------------------------------------------
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    pytesseract \
    ultralytics \
    rembg \
    onnxruntime \
    werkzeug

# --------------------------------------------------
# Expose port
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Start server
# --------------------------------------------------
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
