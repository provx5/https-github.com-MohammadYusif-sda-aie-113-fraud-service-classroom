"""Batch scoring entry point."""

import logging
from pathlib import Path

import pandas as pd

from fraud_service.adapters.sklearn_model import SklearnModel
from fraud_service.domain.entities import Transaction
from fraud_service.domain.policies import FraudPolicy
from fraud_service.service.scorer import FraudScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Load model, score transactions from CSV, write results."""
    
    # Find data and model paths relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    data_path = repo_root / "data" / "transactions_sample.csv"
    model_path = repo_root / "models" / "fraud_xgb_v3.joblib"
    output_path = repo_root / "scored.csv"
    
    logger.info(f"Loading model from {model_path}")
    model = SklearnModel.from_joblib(str(model_path))
    logger.info(f"Loaded model version: {model.version}")
    
    logger.info(f"Loading transactions from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Create scorer with default policy
    policy = FraudPolicy.default()
    scorer = FraudScorer(model, policy)
    
    # Score each transaction
    logger.info("Scoring transactions...")
    results = []
    for _, row in df.iterrows():
        raw_tx = Transaction(**row.to_dict())
        scored = scorer.score(raw_tx)
        results.append(scored.model_dump())
    
    # Write results
    scored_df = pd.DataFrame(results)
    scored_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(scored_df)} scores to {output_path}")
    
    # Summary
    print("\n=== SCORING SUMMARY ===")
    print(scored_df["decision"].value_counts().to_string())
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
