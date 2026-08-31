"""Scoring service orchestration."""

import logging
from typing import ClassVar, Protocol

from fraud_service.domain.entities import (
    EnrichedTransaction,
    FraudScore,
    Transaction,
    enrich_transaction,
)
from fraud_service.domain.policies import FraudPolicy

logger = logging.getLogger(__name__)


class Model(Protocol):
    """Protocol for fraud prediction models."""
    
    def predict_proba(self, features) -> list[list[float]]:
        """Predict fraud probability.
        
        Args:
            features: Feature matrix with shape (n_samples, n_features)
        
        Returns:
            Probability matrix where column 1 is fraud probability.
        """
        ...


class FraudScorer:
    """Orchestrates fraud scoring for transactions."""

    FEATURE_COLS: ClassVar[list[str]] = [
        "amount_log",
        "channel",
        "mcc",
        "hour_of_day",
        "is_night",
    ]
    
    def __init__(self, model: Model, policy: FraudPolicy):
        """Initialize scorer with a model and policy.
        
        Args:
            model: Fraud prediction model implementing Model protocol
            policy: Fraud decision policy (thresholds and rules)
        """
        self.model = model
        self.policy = policy
    
    def score(self, raw_transaction: Transaction) -> FraudScore:
        """Score a single transaction.
        
        Args:
            raw_transaction: Raw transaction from CSV
        
        Returns:
            FraudScore with computed score and decision
        """
        enriched = enrich_transaction(raw_transaction)
        score = self._get_score(enriched)
        decision = self.policy.decide(score)
        
        return FraudScore(
            transaction_id=raw_transaction.transaction_id,
            score=score,
            decision=decision,
        )
    
    def _get_score(self, enriched: EnrichedTransaction) -> float:
        """Get fraud probability from the model for an enriched transaction.
        
        Returns 0.0 (allow) if prediction fails, to fail safely.
        """
        try:
            features_dict = enriched.model_dump()
            features_df = __import__("pandas").DataFrame([features_dict])[self.FEATURE_COLS]
            proba_matrix = self.model.predict_proba(features_df)
            fraud_prob = proba_matrix[0][1]
            return float(fraud_prob)
        except Exception as e:  # noqa: BLE001 - fail closed for model issues
            logger.warning(f"Score prediction failed for {enriched.transaction_id}: {e}")
            return 0.0
