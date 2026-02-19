# ROL Y CONTEXTO
Actúa como un **Senior Python Engineer y MLOps Specialist**. Eres experto en Computer Vision, arquitecturas escalables y buenas prácticas de desarrollo en Python.

# OBJETIVO DEL PROYECTO
Tu tarea es generar el código fuente completo y funcional para el proyecto **"Fraud Face"**: un backend en Python diseñado para la prevención de fraude biométrico.
El sistema debe procesar imágenes faciales, vectorizarlas usando AWS Rekognition, realizar análisis por regiones y ejecutar un pipeline de clustering temporal (diario/semanal/mensual) con persistencia en MongoDB.

# RESTRICCIONES TÉCNICAS (STRICT)
- **Lenguaje:** Python 3.11+
- **Tipado:** Estricto (`typing` module), validado con Pydantic V2.
- **Estilo:** Black/Ruff, Logging estructurado (JSON), manejo de errores robusto (no bare exceptions).
- **Infraestructura:** Docker Compose para MongoDB.
- **AWS:** Boto3 con manejo de reintentos (backoff) y timeouts.
- **Librerías Clave:** `boto3`, `pymongo`, `pydantic`, `numpy`, `pillow`, `hdbscan`, `structlog`.

# ENTREGABLES ESPERADOS
Debes generar los archivos necesarios para cumplir con la siguiente estructura y funcionalidad:
1.  **Configuración:** `Makefile`, `pyproject.toml`, `.env.example`, `docker-compose.yml`.
2.  **Core:** Clientes AWS optimizados, utilidades de imagen (procesamiento en memoria).
3.  **Modelos:** DTOs con Pydantic y Esquemas de Base de Datos.
4.  **Lógica de Negocio:**
    *   Preprocesamiento (recorte, normalización, regiones).
    *   Vectorización (AWS y local).
    *   Pipeline (Ingesta síncrona y enriquecimiento asíncrono).
    *   Clustering y Promoción Temporal.
5.  **Tests:** Pruebas unitarias básicas para los componentes críticos.

---

# ESPECIFICACIONES DETALLADAS Y REGLAS DE NEGOCIO

## 1. Arquitectura y Carpetas
Sigue estrictamente esta estructura de directorios:
```text
fraud-face/
  ├─ Makefile, pyproject.toml, .env.example, docker-compose.yml
  ├─ src/
  │   ├─ app.py                         # Entrypoint (CLI o FastAPI)
  │   ├─ config.py                      # Carga de variables de entorno y validación
  │   ├─ aws/                           # Clientes Boto3 (rekognition_client.py, face_collections.py)
  │   ├─ utils/                         # Logging y helpers (images.py)
  │   ├─ models/                        # DTOs (dto.py)
  │   ├─ storage/                       # Mongo client (mongo.py) y schemas (schemas.py)
  │   ├─ preprocessing/                 # Lógica de imagen (crop.py, normalize.py, regions.py)
  │   ├─ vectorization/                 # Interfaces (backend.py) y implementaciones (aws_rekognition.py, local_embeddings.py)
  │   ├─ clustering/                    # Lógica HDBSCAN (hdbscan_cluster.py) y fusión (merge_agglomerative.py)
  │   └─ pipeline/                      # Ingesta (ingest_sync.py), enriquecimiento (enrich_async.py), webhooks.py, promote_levels.py
  └─ tests/                             # Tests unitarios
```

## 2. Flujo de Datos y Pipeline
### A. Ingesta Síncrona (`src/pipeline/ingest_sync.py`)
1.  Recibir imagen (bytes) + metadatos (original_filename, user_ref).
2.  **Detectar:** Obtener BBox y Landmarks con AWS Rekognition.
3.  **Procesar:** Recortar rostro -> Normalizar a 160x160 (para FaceNet).
4.  **Vectorizar (PyTorch):** Generar embedding (512 dims) usando `facenet-pytorch`.
5.  **Buscar (Mongo Vector Search):** Ejecutar búsqueda vectorial KNN en `faces_daily`.
    *   *Regla:* Si la distancia es menor a `THRESH_DAILY` (similitud alta), marcar `suspicious=True`.
    *   *Regla:* Retornar `original_filename` de los vecinos más cercanos.
6.  **Persistir:** Guardar metadatos y embedding en colección diaria.
    *   Guardar: `image_id`, `original_filename`, `embedding` (512 floats), `bbox`, `landmarks`, `level="daily"`, `created_at`.

### B. Enriquecimiento Asíncrono y Clustering (`src/pipeline/enrich_async.py`, `hdbscan_cluster.py`)
1.  Para rostros nuevos en `faces_daily`, calcular embeddings locales (si el backend local está activo).
2.  Ejecutar **HDBSCAN** sobre los embeddings del día (parámetros configurables).
3.  Asignar `cluster_id` a los documentos.
4.  Calcular centroides (promedio de vectores) y registrar evento `daily_cluster`.

### C. Promoción Temporal (`src/pipeline/promote_levels.py`)
Implementa la lógica de promoción de niveles (Daily → Weekly → Monthly → Yearly):
*   **Comparación:** Usar similitud coseno entre centroides de clusters diarios y clusters existentes en semanal.
*   **Umbrales (Configurables en .env):**
    *   Diario → Semanal: `THRESH_WEEKLY` (ej. 0.70)
    *   Semanal → Mensual: `THRESH_MONTHLY` (ej. 0.65)
    *   Mensual → Anual: `THRESH_YEARLY` (ej. 0.60)
*   **Acción:**
    *   Si similitud >= umbral: Fusionar (añadir miembros al cluster existente en nivel superior).
    *   Si no: Crear nuevo cluster en el nivel superior con los datos actuales.

## 3. Base de Datos (Mongo)
*   **Colecciones:** `faces_daily`, `faces_weekly`, `faces_monthly`, `faces_yearly`, `events`.
*   **Índices:** Obligatorios en `face_id`, `created_at`, `cluster_id`, `user_ref`.
*   **TTL:** Configurar expiración automática en `faces_daily` (ej. 3 días) para limpieza automática.

---

# REFERENCIAS DE IMPLEMENTACIÓN (CÓDIGO BASE)
Usa los siguientes fragmentos como guía lógica, pero **mejora su robustez, manejo de errores y tipado** al implementarlos. Asegúrate de usar `pydantic` settings para la configuración.

```python
# src/aws/rekognition_client.py
import boto3
from botocore.config import Config
def get_rekognition():
    return boto3.client(
        "rekognition",
        config=Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=0.8, connect_timeout=0.5)
    )

# src/preprocessing/crop.py
from PIL import Image
import io
def detect_face_and_bbox(image_bytes: bytes, rekognition):
    resp = rekognition.detect_faces(Image={'Bytes': image_bytes}, Attributes=['ALL'])
    faces = resp.get('FaceDetails', [])
    if not faces:
        raise ValueError("No face detected")
    face = max(faces, key=lambda f: f['Confidence'])
    return face['BoundingBox'], face.get('Landmarks', [])
def crop_bbox(image_bytes: bytes, bbox):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    l = int(bbox['Left'] * w); t = int(bbox['Top'] * h)
    r = int((bbox['Left'] + bbox['Width']) * w)
    b = int((bbox['Top'] + bbox['Height']) * h)
    return img.crop((l, t, r, b))

# src/preprocessing/regions.py
from PIL import Image
def crop_central_region(full_img: Image.Image, landmarks: list):
    lm = {l['Type']: (l['X'], l['Y']) for l in landmarks}
    w, h = full_img.size
    xs, ys = [], []
    keys = ['eyeLeft','eyeRight','nose','leftEyeBrowLeft','leftEyeBrowRight','rightEyeBrowLeft','rightEyeBrowRight','leftEarTragus','rightEarTragus']
    for k in keys:
        if k in lm:
            xs.append(lm[k][0]*w); ys.append(lm[k][1]*h)
    if not xs:
        # Fallback si no hay landmarks suficientes
        x0,y0,x1,y1 = int(0.2*w), int(0.2*h), int(0.8*w), int(0.7*h)
    else:
        m = 0.1 # Margen
        x0 = max(int(min(xs) - m*w), 0); y0 = max(int(min(ys) - m*h), 0)
        x1 = min(int(max(xs) + m*w), w); y1 = min(int(max(ys) + m*h), h)
    return full_img.crop((x0, y0, x1, y1))

# src/preprocessing/normalize.py
from PIL import Image, ImageOps
def normalize(img: Image.Image, size: int = 320):
    img = ImageOps.exif_transpose(img)
    return img.resize((size, size))

# src/vectorization/backend.py
from typing import Protocol, Tuple, Optional, Any, List
class VectorBackend(Protocol):
    def embed_region(self, region_bytes: bytes) -> List[float]:
        ...

# src/vectorization/local_embeddings.py
# Implementación con DeepFace (Facenet512)
import io
import numpy as np
from PIL import Image
from typing import List
from .backend import VectorBackend

# Importación diferida o condicional
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

class LocalEmbeddingBackend(VectorBackend):
    def embed_region(self, region_bytes: bytes) -> List[float]:
        if DeepFace is None:
             raise ImportError("deepface library required")
        
        # Convertir bytes a imagen PIL RGB
        img = Image.open(io.BytesIO(region_bytes)).convert("RGB")
        img_np = np.array(img) # Deepface consume numpy array (RGB/BGR)

        # Generar embedding con Facenet512 (512 dims)
        # enforce_detection=False porque ya detectamos con AWS
        embedding_objs = DeepFace.represent(
            img_path=img_np, 
            model_name="Facenet512",
            enforce_detection=False,
            align=False
        )
        if not embedding_objs:
            return []
            
        vector = embedding_objs[0]["embedding"]
        return vector # Lista de 512 floats


# src/storage/mongo.py
from pymongo import MongoClient, ASCENDING, IndexModel
import os
def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client[os.getenv("MONGO_DB","fraud_face")]

def ensure_indexes(db):
    # Índices estándar
    for c in ["faces_daily","faces_weekly","faces_monthly","faces_yearly","events"]:
        db[c].create_index([("created_at", ASCENDING)])
        db[c].create_index([("cluster_id", ASCENDING)])
    
    db["faces_daily"].create_index("created_at", expireAfterSeconds=3*24*3600)

    # NOTA: Los índices vectoriales (Atlas Vector Search) se definen usualmente
    # en la configuración de Atlas, no vía pymongo create_index estándar,
    # pero aquí simulamos la estructura mental.
    # Definition for Atlas Search Index would be JSON-based.

# src/pipeline/ingest_sync.py
import os, uuid
# imports omitidos...
def ingest_sync(image_bytes: bytes, original_filename: str, user_ref: str | None = None):
    db = get_db(); rek = get_rekognition()
    
    # 1. Detectar (AWS)
    bbox, landmarks = detect_face_and_bbox(image_bytes, rek)
    
    # 2. Procesar (Local)
    face_crop = crop_bbox(image_bytes, bbox)
    face_norm = normalize(face_crop, size=int(os.getenv("IMG_SIZE","320")))
    region_crop = crop_central_region(face_norm, landmarks) # Recorte de características clave
    
    # 3. Vectorizar (Local)
    # Suponiendo conversion de region_crop a bytes para el backend
    # buffer = io.BytesIO(); region_crop.save(buffer, format="JPEG"); region_bytes = buffer.getvalue()
    backend = LocalEmbeddingBackend() 
    embedding = backend.embed_region(region_bytes) # Retorna lista de floats

    # 4. Buscar (Mongo Vector Search)
    # Ejemplo conceptual usando pipeline de agregación para vector search
    # matches = vector_search(db, "faces_daily", embedding, top_k=5)
    
    # 5. Persistir
    image_id = str(uuid.uuid4())
    doc = {
        "image_id": image_id,
        "original_filename": original_filename,
        "embedding": embedding, # Vector para búsquedas futuras
        "bbox": [bbox['Left'], bbox['Top'], bbox['Width'], bbox['Height']],
        "landmarks": landmarks, 
        "level": "daily",
        "created_at": __import__("datetime").datetime.utcnow(),
        "user_ref": user_ref,
    }
    db["faces_daily"].insert_one(doc)
    
    return {"suspicious": False, "matches": [], "doc_id": image_id}
```

---

# INSTRUCCIONES DE EJECUCIÓN
1.  Analiza la estructura de carpetas requerida y la arquitectura propuesta.
2.  Genera primero los archivos de configuración (`pyproject.toml`, `.env.example`, `Makefile`) y modelos (`dto.py`, `config.py`).
3.  Implementa las utilidades de AWS (`rekognition_client.py`) y Preprocesamiento.
4.  Implementa la capa de almacenamiento (`mongo.py`) y vectorización.
5.  Desarrolla la lógica del pipeline (`ingest_sync`, `enrich`) y clustering.
6.  Finaliza con el `main` entrypoint, Docker Compose y los tests.
