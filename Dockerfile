FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt requirements-space.txt

RUN pip install --upgrade pip && \
    pip install -r requirements-space.txt

COPY . .

RUN python data/loader.py && \
    python data/drift_injector.py && \
    python models/lgbm_model.py && \
    python models/isolation_forest.py && \
    python models/autoencoder.py && \
    python models/lstm_forecast.py && \
    python models/champion_challenger.py && \
    rm -f "data/default of credit card clients.xls"

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
