FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .

RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

COPY src/ ./src/
COPY data/ ./data/
COPY dvc.yaml .

CMD ["python", "-m", "src.glucotwin.models.baselines"]
