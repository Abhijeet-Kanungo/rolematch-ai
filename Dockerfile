FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, frontend, and precomputed embedding index
COPY app ./app
COPY static ./static
COPY job_embeddings.npz ./job_embeddings.npz
COPY eval.py ./eval.py

# Render and most PaaS clouds inject $PORT dynamically (defaults to 8000 locally)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]