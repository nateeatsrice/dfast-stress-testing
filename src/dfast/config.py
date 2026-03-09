"""Application configuration via Pydantic BaseSettings with environment variable overrides."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the DFAST stress-testing pipeline.

    All values can be overridden by environment variables with the same name
    (case-insensitive). For example, ``export S3_BUCKET_RAW=my-bucket``.
    """

    # ── S3 Buckets ───────────────────────────────────────────
    S3_BUCKET_RAW: str = "dfast-raw"
    S3_BUCKET_FEATURES: str = "dfast-features"
    S3_BUCKET_MODELS: str = "dfast-models"
    S3_BUCKET_MONITORING: str = "dfast-monitoring"

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://localhost:5432/dfast"

    # ── MLflow ───────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # ── Local Development ────────────────────────────────────
    USE_LOCAL_STORAGE: bool = True
    LOCAL_DATA_DIR: str = "./data"

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
