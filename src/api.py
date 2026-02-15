from __future__ import annotations

import contextlib
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aws.face_collections import ensure_required_collections
from aws.rekognition_client import get_rekognition_client
from config import get_settings, Settings
from models.dto import IngestResultDTO
from pipeline.enrich_async import enrich_daily_faces
from pipeline.ingest_sync import ingest_sync
from pipeline.promote_levels import (
    promote_daily_to_weekly,
    promote_monthly_to_yearly,
    promote_weekly_to_monthly,
)
from storage.mongo import MongoStorage, get_db
from utils.logging import configure_logging

# --- Dependencies & Lifespan ---

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    configure_logging(settings)
    yield
    # Shutdown (if needed)

def get_app_settings() -> Settings:
    return get_settings()

def get_storage(settings: Annotated[Settings, Depends(get_app_settings)]) -> MongoStorage:
    return MongoStorage(get_db(settings), settings)

app = FastAPI(
    title="Fraud Face Backend API",
    description="API REST para prevención de fraude biométrico",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Validations Models ---

class GenericResponse(BaseModel):
    message: str
    details: dict | None = None

# --- Endpoints ---

@app.post("/ingest", response_model=IngestResultDTO)
async def ingest_image(
    file: Annotated[UploadFile, File(...)],
    storage: Annotated[MongoStorage, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    filename: str | None = None,
    user_ref: str | None = None,
):
    """
    Procesa e ingesta una imagen.
    Retorna si es sospechosa y los matches encontrados.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    actual_filename = filename or file.filename
    
    try:
        content = await file.read()
        
        # ingest_sync es async
        result = await ingest_sync(
            image_bytes=content,
            original_filename=actual_filename,
            user_ref=user_ref,
            settings=settings,
            storage=storage,
        )
        return result

    except Exception as e:
        # En producción, loggear el error real
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enrich", response_model=GenericResponse)
async def trigger_enrich(
    background_tasks: BackgroundTasks,
    storage: Annotated[MongoStorage, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_app_settings)],
):
    """
    Calcula embeddings locales y clustering diario (Background Task).
    """
    # enrich_daily_faces es sincrónica, la ejecutamos en background thread pool
    background_tasks.add_task(enrich_daily_faces, settings=settings, storage=storage)
    return GenericResponse(message="Enrich process started in background")


@app.post("/promote/{level}", response_model=GenericResponse)
async def trigger_promote(
    level: str,
    background_tasks: BackgroundTasks,
    storage: Annotated[MongoStorage, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_app_settings)],
):
    """
    Promueve clusters temporalmente.
    Level: 'weekly', 'monthly', 'yearly'
    """
    if level == "weekly":
        background_tasks.add_task(promote_daily_to_weekly, settings=settings, storage=storage)
    elif level == "monthly":
        background_tasks.add_task(promote_weekly_to_monthly, settings=settings, storage=storage)
    elif level == "yearly":
        background_tasks.add_task(promote_monthly_to_yearly, settings=settings, storage=storage)
    else:
        raise HTTPException(status_code=400, detail="Invalid level. Use: weekly, monthly, yearly")

    return GenericResponse(message=f"Promote process for {level} started in background")


@app.post("/system/ensure-indexes", response_model=GenericResponse)
async def ensure_indexes(
    storage: Annotated[MongoStorage, Depends(get_storage)],
):
    """
    Crea índices requeridos en MongoDB.
    """
    storage.ensure_indexes()
    return GenericResponse(message="Indexes ensured successfully")


@app.post("/system/ensure-collections", response_model=GenericResponse)
async def ensure_collections(
    settings: Annotated[Settings, Depends(get_app_settings)],
):
    """
    Crea colecciones de Rekognition.
    """
    # get_rekognition_client es async
    rekognition = get_rekognition_client(settings)
    ensure_required_collections(rekognition, settings)
    return GenericResponse(message="Collections ensured successfully")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
