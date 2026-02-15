# Fraud Face

Backend en Python para prevención de fraude biométrico mediante vectores faciales, clustering temporal (HDBSCAN) y persistencia en MongoDB. Este sistema utiliza AWS Rekognition para generar embeddings faciales robustos y gestiona la evolución temporal de identidades mediante un pipeline de promoción por niveles (diario, semanal, mensual, anual).

## Características Principales

*   **Vectorización Facial**: Integración con AWS Rekognition para obtener embeddings de alta calidad.
*   **Clustering Avanzado**: Uso de HDBSCAN para agrupar rostros similares y detectar identidades recurrentes.
*   **Pipeline Temporal**: Estrategia de promoción de clusters (Diario -> Semanal -> Mensual -> Anual) para gestión eficiente de identidades a largo plazo.
*   **Almacenamiento NoSQL**: Persistencia flexible de metadatos y vectores en MongoDB.
*   **CLI Unificada**: Herramienta de línea de comandos para ingestión, enriquecimiento y mantenimiento.

## Tecnologías

*   **Lenguaje**: Python 3.11+
*   **Cloud**: AWS Rekognition
*   **Base de Datos**: MongoDB (pymongo)
*   **Ciencia de Datos**: NumPy, HDBSCAN, Scikit-learn
*   **Validación**: Pydantic
*   **Gestión de Dependencias**: pip / pyproject.toml

## Requisitos Previos

*   Python 3.11 o superior.
*   Docker y Docker Compose (para ejecutar MongoDB local).
*   Cuenta de AWS con permisos para Rekognition (access key/secret key).
*   (Opcional) Make para ejecutar comandos de desarrollo.

## Instalación

1.  **Clonar el repositorio**:
    ```bash
    git clone <url-del-repo>
    cd fraud-face
    ```

2.  **Crear y activar entorno virtual**:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instalar dependencias**:
    ```bash
    # Producción
    pip install -e .
    
    # Desarrollo (incluye tests, linting)
    pip install -e .[dev]
    
    # O usando Make
    make install-dev
    ```

## Configuración

Crea un archivo `.env` en la raíz del proyecto basándote en el siguiente ejemplo:

```ini
APP_ENV=dev
LOG_LEVEL=INFO

# Configuración AWS
AWS_REGION=us-east-1
# AWS_PROFILE=default  # Opcional si usas perfil nombrado
# AWS_ACCESS_KEY_ID=... # Opcional si no usas perfil/roles
# AWS_SECRET_ACCESS_KEY=...

# Timeouts AWS Rekognition
AWS_REKOGNITION_MAX_ATTEMPTS=3
AWS_CONNECT_TIMEOUT=0.5
AWS_READ_TIMEOUT=0.8

# Nombres de Colecciones Rekognition
REKOGNITION_COLLECTION_DAILY=faces-daily
REKOGNITION_COLLECTION_WEEKLY=faces-weekly
REKOGNITION_COLLECTION_MONTHLY=faces-monthly
REKOGNITION_COLLECTION_YEARLY=faces-yearly

# Configuración MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=fraud_face
```

## Uso

### Levantar Infraestructura Local (MongoDB)

```bash
docker-compose up -d
```

### Inicialización

Antes de usar el sistema, asegúrate de crear los índices en Mongo y las colecciones en AWS si no existen.

```bash
# Crear índices en MongoDB
python -m src.app ensure-indexes

# Crear colecciones en AWS Rekognition
python -m src.app ensure-collections
```

### Comandos CLI

El punto de entrada principal es `src/app.py` (o `make run` como atajo).

#### 1. Ingestar una Imagen
Procesa una imagen local, detecta rostros y guarda los vectores.

```bash
python -m src.app ingest --image "ruta/a/imagen.jpg" --user-ref "usuario-123"
```

#### 2. Enriquecimiento (Clustering)
Ejecuta el algoritmo HDBSCAN sobre los datos ingeridos para formar clusters diarios.

```bash
python -m src.app enrich
```

#### 3. Promoción Temporal
Mueve clusters consolidados a niveles superiores (ej. de diario a semanal).

```bash
python -m src.app promote
```

## Desarrollo

El proyecto incluye un `Makefile` para tareas comunes:

*   `make format`: Formatea el código con Black y Ruff.
*   `make lint`: Analiza el código en busca de errores con Ruff.
*   `make typecheck`: Verifica tipos estáticos con MyPy.
*   `make test`: Ejecuta pruebas unitarias con Pytest.

## Estructura del Proyecto

```
src/
├── app.py             # Punto de entrada CLI
├── aws/               # Clientes y utilidades AWS
├── clustering/        # Lógica de HDBSCAN y fusión
├── models/            # DTOs y esquemas de datos
├── pipeline/          # Flujos de ingestión y promoción
├── preprocessing/     # Manipulación de imágenes
├── storage/           # Repositorios MongoDB
├── utils/             # Logging y helpers
└── vectorization/     # Interfaces de vectorización
```
