# ==============================================================================
# DocShield AI — Production Multi-Stage Secure Dockerfile
# ==============================================================================

FROM python:3.12-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

# Install system runtime dependencies for Tesseract, pyzbar, and headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for zero-root execution
RUN groupadd -r docshield && useradd -r -g docshield -m -d /home/docshield docshield

WORKDIR /app

# Install PyTorch CPU wheel first (150MB instead of 2.8GB GPU wheel, preventing build timeouts and OOM)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create secure instance and upload directories with correct user ownership
RUN mkdir -p instance uploads app/ml/weights && \
    chown -R docshield:docshield /app

# Switch to unprivileged user (Defense against Container Escapes)
USER docshield

# Expose standard API port
EXPOSE 5000

# Healthcheck validating API availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/api/v1/health || exit 1

# Production WSGI server execution via Gunicorn (tuned for low-memory container platforms like Render)
# --preload: Loads app and ML models once before forking, saving RAM and eliminating per-worker cold start
# --threads 2: Reduced from 4 to lower memory pressure on Render's 512MB free tier
# --timeout 120: Provides sufficient headroom for cold start without premature worker termination
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 --preload wsgi:app"]
