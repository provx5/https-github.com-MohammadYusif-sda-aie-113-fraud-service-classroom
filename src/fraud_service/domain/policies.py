"""Business policies for fraud decision-making."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FraudPolicy:
    """Encapsulates the decision thresholds and rules."""
    block_threshold: float
    review_threshold: float
    
    @staticmethod
    def default() -> "FraudPolicy":
        """Default policy: block at 0.85, review 0.15 points below that."""
        return FraudPolicy(
            block_threshold=0.85,
            review_threshold=0.85 - 0.15,
        )
    
    def decide(self, score: float) -> Literal["block", "review", "allow"]:
        """Convert a fraud probability score to a decision."""
        if score >= self.block_threshold:
            return "block"
        elif score >= self.review_threshold:
            return "review"
        else:
            return "allow"
