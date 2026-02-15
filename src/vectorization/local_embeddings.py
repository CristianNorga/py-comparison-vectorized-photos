import numpy as np
import io
from PIL import Image
from typing import List
from vectorization.backend import VectorBackend

class LocalEmbeddingBackend(VectorBackend):
    def embed_region(self, region_bytes: bytes) -> List[float]:
        # Implementar lógica de inferencia local real aquí.
        # Por ahora simulamos un embedding dummy.
        try:
            pil_img = Image.open(io.BytesIO(region_bytes)).convert("L").resize((16, 16))
            arr = np.asarray(pil_img, dtype=np.float32).flatten()
            norm = np.linalg.norm(arr)
            if norm == 0:
                vector = arr
            else:
                vector = arr / norm
            
            # Asegurar que sea una lista de floats serializable
            return vector.tolist()
        except:
            # Fallback en caso de error de imagen
            return [0.0] * 256
