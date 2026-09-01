from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the fraud service."""

    model_config = SettingsConfigDict(env_prefix="FRAUD_", extra="ignore")

    model_path: Path = Field(
        default=Path("models/fraud_xgb_v3.joblib"),
        description="Path to the trained model bundle.",
    )
    log_level: str = Field(default="INFO", description="Logging level for the application.")
    data_path: Path = Field(default=Path("data/transactions_sample.csv"), description="Path to the CSV dataset.")

    @field_validator("model_path", "data_path")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        return value if value.is_absolute() else (Path.cwd() / value).resolve()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper


settings = Settings()
