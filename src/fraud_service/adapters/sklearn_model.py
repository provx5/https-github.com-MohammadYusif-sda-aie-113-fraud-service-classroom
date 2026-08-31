"""Adapters for external dependencies (sklearn, joblib)."""

from typing import Protocol

import joblib


class Model(Protocol):
    """Protocol for fraud prediction models."""
    
    def predict_proba(self, features) -> list[list[float]]:
        """Predict fraud probability.
        
        Args:
            features: Feature matrix with shape (n_samples, n_features)
        
        Returns:
            Probability matrix with shape (n_samples, 2) where column 1 is fraud prob.
        """
        ...


class SklearnModel:
    """Wrapper for scikit-learn pipeline models loaded from joblib."""
    
    def __init__(self, pipeline, version: str):
        """Initialize with a scikit-learn pipeline.
        
        Args:
            pipeline: Fitted sklearn pipeline
            version: Model version string
        """
        self.pipeline = pipeline
        self.version = version
    
    def predict_proba(self, features) -> list[list[float]]:
        """Predict fraud probability using the pipeline."""
        return self.pipeline.predict_proba(features).tolist()
    
    @staticmethod
    def from_joblib(path: str) -> "SklearnModel":
        """Load a model from a joblib bundle file.
        
        The bundle must be a dict with keys:
        - "pipeline": the fitted sklearn pipeline
        - "version": version string
        """
        bundle = joblib.load(path)
        return SklearnModel(
            pipeline=bundle["pipeline"],
            version=bundle["version"],
        )
