FROM python:3.12-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip setuptools wheel build
COPY pyproject.toml .
COPY src ./src
RUN python -m build

FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

RUN groupadd --system app && useradd --system --gid app --create-home app && \
    apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

COPY . /app
RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/v1/ready || exit 1

CMD ["uvicorn", "fraud_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
