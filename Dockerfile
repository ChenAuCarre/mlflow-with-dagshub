FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
# Repo is public, so no token is needed to read. If it goes private, pass
# MLFLOW_TRACKING_USERNAME/PASSWORD here via a BuildKit secret (not ARG/ENV).
RUN python fetch_model.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
