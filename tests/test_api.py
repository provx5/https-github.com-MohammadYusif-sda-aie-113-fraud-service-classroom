from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_service.adapters.sklearn_model import SklearnModel
from fraud_service.api.app import create_app
from fraud_service.api.routes import get_scorer
from fraud_service.domain.entities import Transaction
from fraud_service.domain.policies import FraudPolicy
from fraud_service.service.scorer import FraudScorer


class FixedScorer:
    def score(self, raw_transaction):
        return type(
            "Result",
            (),
            {"transaction_id": raw_transaction.transaction_id, "score": 0.27, "decision": "review"},
        )()


@pytest.fixture
def api_client():
    app = create_app()
    app.dependency_overrides[get_scorer] = lambda: FixedScorer()
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_api_predict_success(api_client):
    payload = {
        "transaction_id": "TXN-API-0001",
        "amount_sar": 42.67,
        "channel": "pos",
        "merchant_category": "healthcare",
        "customer_id": "CUST-0001",
        "timestamp": "2026-07-18T03:43:47Z",
        "ip_country": "US",
        "device_type": "mobile",
    }

    response = api_client.post("/v1/predict", json=payload, headers={"X-Trace-Id": "trace-123"})

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "trace-123"
    body = response.json()
    assert body["transaction_id"] == payload["transaction_id"]
    assert body["decision"] == "review"
    assert 0.0 <= body["score"] <= 1.0
    assert body["model_version"]


@pytest.mark.integration
def test_api_validates_bad_payloads(api_client):
    bad_payload = {
        "transaction_id": "",
        "amount_sar": -10,
        "channel": "pos",
        "merchant_category": "healthcare",
        "customer_id": "CUST-0001",
        "timestamp": "2026-07-18T03:43:47Z",
        "ip_country": "US",
        "device_type": "mobile",
    }

    response = api_client.post("/v1/predict", json=bad_payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["trace_id"]
    assert response.headers["X-Trace-Id"] == body["error"]["trace_id"]


@pytest.mark.integration
def test_api_health_and_ready():
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/v1/health")
        ready = client.get("/v1/ready")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.json()["status"] == "ready"


@pytest.mark.integration
def test_real_model_matches_golden_file():
    model = SklearnModel.from_joblib("models/fraud_xgb_v3.joblib")
    scorer = FraudScorer(model, FraudPolicy.default())
    golden = pd.read_csv("data/golden_scores_v3.csv")

    scores = []
    for _, row in golden.iterrows():
        tx = Transaction(
            transaction_id=row["transaction_id"],
            amount_sar=float(row["amount_sar"]),
            channel=str(row["channel"]),
            merchant_category=str(row["mcc"]),
            customer_id=str(row["customer_id"]),
            timestamp=str(row["timestamp"]),
            ip_country="US",
            device_type="mobile",
        )
        scores.append(scorer.score(tx).score)

    assert scores == pytest.approx(golden["score"].tolist(), abs=1e-9)


@pytest.mark.integration
def test_real_model_is_invariant_to_case_in_merchant_category():
    model = SklearnModel.from_joblib("models/fraud_xgb_v3.joblib")
    scorer = FraudScorer(model, FraudPolicy.default())

    lower = Transaction(
        transaction_id="TXN-CASE-LOWER",
        amount_sar=25.0,
        channel="pos",
        merchant_category="healthcare",
        customer_id="CUST-0001",
        timestamp="2026-07-18T03:43:47Z",
        ip_country="US",
        device_type="mobile",
    )
    upper = Transaction(
        transaction_id="TXN-CASE-UPPER",
        amount_sar=25.0,
        channel="pos",
        merchant_category="HEALTHCARE",
        customer_id="CUST-0001",
        timestamp="2026-07-18T03:43:47Z",
        ip_country="US",
        device_type="mobile",
    )

    assert scorer.score(lower).score == pytest.approx(scorer.score(upper).score, abs=1e-12)


@pytest.mark.integration
def test_real_model_score_increases_with_larger_amounts():
    model = SklearnModel.from_joblib("models/fraud_xgb_v3.joblib")
    scorer = FraudScorer(model, FraudPolicy.default())

    small = Transaction(
        transaction_id="TXN-SMALL",
        amount_sar=10.0,
        channel="pos",
        merchant_category="HEALTHCARE",
        customer_id="CUST-0001",
        timestamp="2026-07-18T03:43:47Z",
        ip_country="US",
        device_type="mobile",
    )
    large = Transaction(
        transaction_id="TXN-LARGE",
        amount_sar=5000.0,
        channel="pos",
        merchant_category="HEALTHCARE",
        customer_id="CUST-0001",
        timestamp="2026-07-18T03:43:47Z",
        ip_country="US",
        device_type="mobile",
    )

    assert scorer.score(large).score > scorer.score(small).score
