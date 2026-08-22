FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Pre-download the FastEmbed ONNX model during the Docker build process
# so it doesn't download at runtime.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='intfloat/multilingual-e5-small', cache_dir='/app/fastembed_cache')"

# Set environment variables for memory efficiency (even on Cloud Run)
ENV OMP_NUM_THREADS="1"
ENV TOKENIZERS_PARALLELISM="false"
ENV ONNXRUNTIME_MAX_THREADS="1"

# Expose port (Cloud Run sets the PORT env var)
ENV PORT 8080

# Command to run the FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
