import io
import logging
from typing import List

import torch
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None

from vectorization.backend import VectorBackend

logger = logging.getLogger(__name__)

class LocalEmbeddingBackendFaceNet(VectorBackend):
    def __init__(self, model_name: str = "vggface2"): 
        """
        Inicializa FaceNet con PyTorch.
        Classify=False para obtener embeddings en lugar de logits.
        """
        if InceptionResnetV1 is None:
            raise RuntimeError("facenet-pytorch is required. Install with 'pip install facenet-pytorch torch'")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Cargando modelo FaceNet (facenet-pytorch) en {self.device}...")
        
        # Pretrained on VGGFace2 (sota) or CASIA-WebFace
        self.resnet = InceptionResnetV1(pretrained=model_name).eval().to(self.device)

    def embed_region(self, region_bytes: bytes) -> List[float]:
        try:
            # 1. Cargar imagen desde bytes
            img = Image.open(io.BytesIO(region_bytes)).convert("RGB")
            
            # 2. Preprocesamiento manual requerido por InceptionResnetV1
            # Resize a 160x160 (entrada estándar de FaceNet)
            img = img.resize((160, 160))
            
            # Convertir a numpy array
            img_np = np.array(img).astype(np.float32)

            # Normalización (Whiten): (Pixels - 127.5) / 128.0
            img_tensor = torch.tensor(img_np)
            img_tensor = (img_tensor - 127.5) / 128.0
            
            # Permutar dimensiones: [H, W, C] -> [C, H, W] -> [1, C, H, W]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 3. Inferencia
            with torch.no_grad():
                embedding_tensor = self.resnet(img_tensor)
                
            # 4. Retornar lista plana (512 dimensiones)
            vector = embedding_tensor.detach().cpu().numpy()[0].tolist()
            return vector

        except Exception as e:
            logger.error(f"Error generando embedding con FaceNet-PyTorch: {e}")
            return [0.0] * 512
