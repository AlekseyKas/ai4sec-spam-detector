FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py preprocess.py ./
COPY models/ ./models/
ENV PORT=8000 MODEL_DIR=/app/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} app:app"]
