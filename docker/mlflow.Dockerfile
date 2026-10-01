# MLflow tracking server with the Postgres driver preinstalled.
FROM python:3.13-slim
RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.13  psycopg==3.1.18
EXPOSE 5000