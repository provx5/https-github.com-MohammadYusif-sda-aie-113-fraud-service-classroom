FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY models ./models

RUN python -m pip install --no-cache-dir .

CMD ["python", "-m", "fraud_service.batch"]
