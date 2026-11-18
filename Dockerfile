FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HAVEN_DB_PATH=/data/haven_history.sqlite3
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app.py ./
COPY src ./src
COPY model ./model
COPY data ./data
COPY static ./static
RUN mkdir -p logs /data
VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
