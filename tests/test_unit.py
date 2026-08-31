import pytest

from fraud_service.domain.entities import Transaction, enrich_transaction
from fraud_service.domain.policies import FraudPolicy
from fraud_service.service.scorer import FraudScorer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "allow"),
        (0.699, "allow"),
        (0.700, "review"),
        (0.849, "review"),
        (0.850, "block"),
    ],
)
def test_policy_boundaries(score, expected):
    policy = FraudPolicy.default()
    assert policy.decide(score) == expected


@pytest.mark.unit
def test_feature_extraction_stabilises_names_and_boundary_hours():
    raw = Transaction(
        transaction_id="TXN-TEST-001",
        amount_sar=42.67,
        channel=" ECom ",
        merchant_category=" health & beauty ",
        customer_id="CUST-0001",
        timestamp="2026-07-05T05:59:59Z",
        ip_country="US",
        device_type="mobile",
    )

    enriched = enrich_transaction(raw)

    assert FraudScorer.FEATURE_COLS == ["amount_log", "channel", "mcc", "hour_of_day", "is_night"]
    assert list(enriched.model_dump().keys()) == [
        "transaction_id",
        "amount_log",
        "channel",
        "mcc",
        "hour_of_day",
        "is_night",
    ]
    assert enriched.channel == "ecom"
    assert enriched.mcc == "HEALTH_&_BEAUTY"
    assert enriched.hour_of_day == 5
    assert enriched.is_night == 1

    daybreak = raw.model_copy(update={"timestamp": "2026-07-05T06:00:00Z"})
    assert enrich_transaction(daybreak).is_night == 0
