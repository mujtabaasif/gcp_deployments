FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

COPY . .

RUN uv sync

EXPOSE 8000

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
