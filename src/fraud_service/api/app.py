from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraud_service.adapters.sklearn_model import SklearnModel
from fraud_service.api.routes import router
from fraud_service.domain.policies import FraudPolicy
from fraud_service.logging_utils import configure_logging
from fraud_service.service.scorer import FraudScorer
from fraud_service.settings import settings

logger = structlog.get_logger(__name__)


def _error_payload(status_code: int, code: str, message: str, trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "trace_id": trace_id}},
        headers={"X-Trace-Id": trace_id},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    model = SklearnModel.from_joblib(str(settings.model_path))
    app.state.model = model
    app.state.scorer = FraudScorer(model, FraudPolicy.default())
    logger.info("model_loaded", model_version=model.version, model_path=str(settings.model_path))
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Fraud Service", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or request.headers.get("x-trace-id") or str(uuid4())
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        trace_id = getattr(request.state, "trace_id", str(uuid4()))
        logger.warning("validation_error", trace_id=trace_id, errors=exc.errors())
        return _error_payload(422, "validation_error", "Request validation failed.", trace_id)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        trace_id = getattr(request.state, "trace_id", str(uuid4()))
        logger.warning("http_exception", trace_id=trace_id, status_code=exc.status_code)
        return _error_payload(exc.status_code, "http_error", exc.detail, trace_id)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", str(uuid4()))
        logger.exception("unhandled_exception", trace_id=trace_id, path=str(request.url.path))
        return _error_payload(500, "internal_error", "An unexpected error occurred.", trace_id)

    app.include_router(router)
    return app


app = create_app()
