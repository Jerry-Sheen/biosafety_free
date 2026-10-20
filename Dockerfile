# ==============================================================================
# Multi-Stage Secure Dockerfile for Biosafety Free Application
# ==============================================================================

FROM python:3.12-slim AS base

# Set environment variables for Python, Streamlit, and ChromaDB
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8601 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    CHROMA_API_IMPL=chromadb.api.segment.SegmentAPI \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Install system dependencies for OCR (Tesseract Korean & English) and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-kor \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (with BuildKit cache)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Create application user and set wide permissions for persistent data volumes
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/chroma_db /app/data/uploads /app/logs && \
    chmod -R 777 /app/data /app/logs && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . .

# Switch to appuser
USER appuser

EXPOSE 8600 8601 8602

CMD ["streamlit", "run", "app.py", "--server.port=8601", "--server.address=0.0.0.0"]
