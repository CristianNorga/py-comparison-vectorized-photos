from botocore.config import Config
import boto3
from botocore.client import BaseClient

from config import Settings


def get_rekognition_client(settings: Settings) -> BaseClient:
    return boto3.client(
        "rekognition",
        region_name=settings.aws_region,
        config=Config(
            retries={"max_attempts": settings.aws_rekognition_max_attempts, "mode": "standard"},
            connect_timeout=settings.aws_connect_timeout,
            read_timeout=settings.aws_read_timeout,
        ),
    )
