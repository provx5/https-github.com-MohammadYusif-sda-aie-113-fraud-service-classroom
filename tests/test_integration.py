import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud_service.domain.entities import Transaction
from fraud_service.domain.policies import FraudPolicy
from fraud_service.service.scorer import FraudScorer


@pytest.mark.integration
def test_valid_transaction_scores_with_expected_contract(scorer):
    raw = Transaction(
        transaction_id="TXN-VALID-001",
        amount_sar=42.67,
        channel="pos",
        merchant_category="healthcare",
        customer_id="CUST-0001",
        timestamp="2026-07-18T03:43:47Z",
        ip_country="US",
        device_type="mobile",
    )

    result = scorer.score(raw)

    assert result.transaction_id == raw.transaction_id
    assert 0.0 <= result.score <= 1.0
    assert result.decision in {"allow", "review", "block"}
    assert result.model_dump()["decision"] == result.decision


@pytest.mark.integration
@pytest.mark.parametrize("payload_path", sorted(Path("payloads/malformed").glob("*.json")))
def test_malformed_payloads_are_rejected(payload_path):
    try:
        payload = json.loads(payload_path.read_text())
    except json.JSONDecodeError:
        return

    with pytest.raises((TypeError, ValueError, ValidationError)):
        Transaction.model_validate(payload)


@pytest.mark.integration
def test_score_failure_fails_safely_without_leaking_exception():
    class ExplodingModel:
        def predict_proba(self, features):
            raise RuntimeError("model exploded")

    raw = Transaction(
        transaction_id="TXN-FAILSAFE-001",
        amount_sar=99.99,
        channel="ecom",
        merchant_category="electronics",
        customer_id="CUST-0002",
        timestamp="2026-07-15T22:05:00Z",
        ip_country="US",
        device_type="desktop",
    )

    scorer = FraudScorer(ExplodingModel(), FraudPolicy.default())
    result = scorer.score(raw)

    assert result.score == 0.0
    assert result.decision == "allow"
