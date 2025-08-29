FROM python:3.10-slim

WORKDIR /app

# system deps - for lightgbm and psycopg2 use build deps if required
RUN apt-get update && apt-get install -y build-essential libpq-dev git ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models/current
EXPOSE 8080 8001

CMD ["gunicorn", "src.serving.api:APP", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "2"]
