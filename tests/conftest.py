import pandas as pd
import pytest

from fraud_service.adapters.sklearn_model import SklearnModel
from fraud_service.domain.policies import FraudPolicy
from fraud_service.service.scorer import FraudScorer


@pytest.fixture(scope="session")
def model():
    return SklearnModel.from_joblib("models/fraud_xgb_v3.joblib")


@pytest.fixture(scope="session")
def sample_transactions():
    return pd.read_csv("data/transactions_sample.csv")


@pytest.fixture
def scorer(model):
    return FraudScorer(model, FraudPolicy.default())
