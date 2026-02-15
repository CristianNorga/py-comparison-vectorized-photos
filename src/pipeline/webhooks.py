from datetime import UTC, datetime
from typing import Any

from storage.mongo import MongoStorage


def register_webhook_event(storage: MongoStorage, event_name: str, payload: dict[str, Any]) -> None:
    storage.insert_event(
        {
            "event_type": "webhook",
            "name": event_name,
            "payload": payload,
            "created_at": datetime.now(UTC),
        }
    )
