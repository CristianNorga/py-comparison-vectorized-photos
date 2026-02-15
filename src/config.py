from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_rekognition_max_attempts: int = Field(default=3, alias="AWS_REKOGNITION_MAX_ATTEMPTS")
    aws_connect_timeout: float = Field(default=0.5, alias="AWS_CONNECT_TIMEOUT")
    aws_read_timeout: float = Field(default=0.8, alias="AWS_READ_TIMEOUT")

    rekognition_collection_daily: str = Field(
        default="faces-daily", alias="REKOGNITION_COLLECTION_DAILY"
    )
    rekognition_collection_weekly: str = Field(
        default="faces-weekly", alias="REKOGNITION_COLLECTION_WEEKLY"
    )
    rekognition_collection_monthly: str = Field(
        default="faces-monthly", alias="REKOGNITION_COLLECTION_MONTHLY"
    )
    rekognition_collection_yearly: str = Field(
        default="faces-yearly", alias="REKOGNITION_COLLECTION_YEARLY"
    )

    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="fraud_face", alias="MONGO_DB")
    mongo_connect_timeout_ms: int = Field(default=3000, alias="MONGO_CONNECT_TIMEOUT_MS")
    mongo_server_selection_timeout_ms: int = Field(
        default=3000, alias="MONGO_SERVER_SELECTION_TIMEOUT_MS"
    )
    faces_daily_ttl_seconds: int = Field(default=259200, alias="FACES_DAILY_TTL_SECONDS")

    img_size: int = Field(default=320, alias="IMG_SIZE")
    top_k_matches: int = Field(default=5, alias="TOP_K_MATCHES")

    thresh_daily: float = Field(default=0.8, alias="THRESH_DAILY")
    thresh_weekly: float = Field(default=0.7, alias="THRESH_WEEKLY")
    thresh_monthly: float = Field(default=0.65, alias="THRESH_MONTHLY")
    thresh_yearly: float = Field(default=0.6, alias="THRESH_YEARLY")

    local_embedding_enabled: bool = Field(default=True, alias="LOCAL_EMBEDDING_ENABLED")
    hdbscan_min_cluster_size: int = Field(default=2, alias="HDBSCAN_MIN_CLUSTER_SIZE")
    hdbscan_min_samples: int = Field(default=1, alias="HDBSCAN_MIN_SAMPLES")

    @field_validator("thresh_daily", "thresh_weekly", "thresh_monthly", "thresh_yearly")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Los umbrales deben estar entre 0 y 1")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
