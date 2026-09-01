from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from fraud_service.api.schemas import PredictRequest, PredictResponse
from fraud_service.domain.entities import Transaction
from fraud_service.service.scorer import FraudScorer

logger = structlog.get_logger(__name__)
router = APIRouter()


def get_scorer(request: Request) -> FraudScorer:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


@router.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fraud-service"}


@router.get("/v1/ready")
def ready(request: Request) -> dict[str, str]:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready", "model_version": getattr(scorer.model, "version", "unknown")}


# ruff: noqa: B008
@router.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request, scorer: FraudScorer = Depends(get_scorer)):
    trace_id = request.headers.get("X-Trace-Id") or request.headers.get("x-trace-id") or str(uuid4())
    request.state.trace_id = trace_id

    transaction_payload = payload.model_dump()
    timestamp = transaction_payload["timestamp"]
    if isinstance(timestamp, str):
        transaction_payload["timestamp"] = timestamp
    else:
        transaction_payload["timestamp"] = timestamp.isoformat().replace("+00:00", "Z")

    raw_tx = Transaction(**transaction_payload)
    score = scorer.score(raw_tx)
    model_version = getattr(getattr(scorer, "model", None), "version", "unknown")
    response = PredictResponse(
        transaction_id=raw_tx.transaction_id,
        score=score.score,
        decision=score.decision,
        model_version=model_version,
    )
    bucket = round(score.score, 2)
    logger.info(
        "prediction_served",
        trace_id=trace_id,
        transaction_id=raw_tx.transaction_id,
        score_bucket=bucket,
        decision=score.decision,
        model_version=model_version,
    )
    return JSONResponse(content=response.model_dump(), headers={"X-Trace-Id": trace_id})
