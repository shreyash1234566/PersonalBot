FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed at runtime
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY data/chromadb/ data/chromadb/
COPY data/examples/ data/examples/

EXPOSE 8080

# Single worker is fine for free tier (512MB RAM)
# Use --limit-max-requests to recycle workers and prevent memory leaks
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 30 --limit-max-requests 1000"]
