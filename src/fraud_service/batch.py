"""Batch scoring entry point."""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
import structlog

from fraud_service.adapters.sklearn_model import SklearnModel
from fraud_service.domain.entities import Transaction
from fraud_service.domain.policies import FraudPolicy
from fraud_service.logging_utils import configure_logging
from fraud_service.service.scorer import FraudScorer
from fraud_service.settings import settings

configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


def main():
    """Load model, score transactions from CSV, write results."""
    model_path = settings.model_path
    data_path = settings.data_path
    output_path = data_path.parent.parent / "scored.csv"

    logger.info("loading_model", model_path=str(model_path))
    model = SklearnModel.from_joblib(str(model_path))
    logger.info("model_loaded", model_version=model.version)

    logger.info("loading_transactions", data_path=str(data_path))
    df = pd.read_csv(data_path)
    logger.info("transactions_loaded", total_rows=len(df))

    scorer = FraudScorer(model, FraudPolicy.default())
    logger.info("scoring_transactions", total_rows=len(df))
    results = []
    for _, row in df.iterrows():
        trace_id = str(uuid4())
        raw_tx = Transaction(**row.to_dict())
        scored = scorer.score(raw_tx)
        bucket = round(scored.score, 2)
        logger.info(
            "prediction_served",
            trace_id=trace_id,
            transaction_id=raw_tx.transaction_id,
            score_bucket=bucket,
            decision=scored.decision,
        )
        results.append(scored.model_dump())

    scored_df = pd.DataFrame(results)
    scored_df.to_csv(output_path, index=False)
    logger.info(
        "scoring_complete",
        output_path=str(output_path),
        total_rows=len(scored_df),
        decision_counts=scored_df["decision"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
