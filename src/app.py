from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aws.face_collections import ensure_required_collections
from aws.rekognition_client import get_rekognition_client
from config import get_settings
from pipeline.enrich_async import enrich_daily_faces
from pipeline.ingest_sync import ingest_sync
from pipeline.promote_levels import (
    promote_daily_to_weekly,
    promote_monthly_to_yearly,
    promote_weekly_to_monthly,
)
from storage.mongo import MongoStorage, get_db
from utils.logging import configure_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Face backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = subparsers.add_parser("ingest", help="Procesa e ingesta una imagen")
    ingest_cmd.add_argument("--image", required=True, help="Ruta del archivo de imagen")
    ingest_cmd.add_argument("--filename", required=False, help="Nombre original")
    ingest_cmd.add_argument("--user-ref", required=False)

    subparsers.add_parser("enrich", help="Calcula embeddings locales y clustering diario")
    subparsers.add_parser("promote", help="Promueve clusters temporalmente")
    subparsers.add_parser("ensure-indexes", help="Crea índices requeridos en MongoDB")
    subparsers.add_parser("ensure-collections", help="Crea colecciones de Rekognition")

    return parser.parse_args()


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=str))


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    args = _parse_args()
    storage = MongoStorage(get_db(settings), settings)

    if args.command == "ensure-indexes":
        storage.ensure_indexes()
        _print({"status": "ok", "action": "ensure-indexes"})
        return

    if args.command == "ensure-collections":
        rekognition = get_rekognition_client(settings)
        ensure_required_collections(rekognition, settings)
        _print({"status": "ok", "action": "ensure-collections"})
        return

    if args.command == "ingest":
        image_path = Path(args.image)
        image_bytes = image_path.read_bytes()
        original_filename = args.filename or image_path.name
        result = ingest_sync(
            image_bytes=image_bytes,
            original_filename=original_filename,
            user_ref=args.user_ref,
            settings=settings,
            storage=storage,
        )
        _print(result.model_dump(mode="json"))
        return

    if args.command == "enrich":
        result = enrich_daily_faces(settings=settings, storage=storage)
        _print(result)
        return

    if args.command == "promote":
        daily_weekly = promote_daily_to_weekly(settings=settings, storage=storage)
        weekly_monthly = promote_weekly_to_monthly(settings=settings, storage=storage)
        monthly_yearly = promote_monthly_to_yearly(settings=settings, storage=storage)
        _print(
            {
                "daily_to_weekly": daily_weekly.model_dump(mode="json"),
                "weekly_to_monthly": weekly_monthly.model_dump(mode="json"),
                "monthly_to_yearly": monthly_yearly.model_dump(mode="json"),
            }
        )
        return

    raise RuntimeError("Comando no soportado")


if __name__ == "__main__":
    main()
