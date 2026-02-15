from botocore.client import BaseClient
from botocore.exceptions import ClientError

from config import Settings
from utils.logging import get_logger

logger = get_logger(__name__)


def ensure_collection(client: BaseClient, collection_name: str) -> None:
    try:
        client.create_collection(CollectionId=collection_name)
        logger.info("rekognition_collection_created", collection=collection_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "ResourceAlreadyExistsException":
            logger.info("rekognition_collection_exists", collection=collection_name)
            return
        raise


def ensure_required_collections(client: BaseClient, settings: Settings) -> None:
    for collection in (
        settings.rekognition_collection_daily,
        settings.rekognition_collection_weekly,
        settings.rekognition_collection_monthly,
        settings.rekognition_collection_yearly,
    ):
        ensure_collection(client, collection)
