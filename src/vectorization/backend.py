from typing import Protocol, List

from PIL import Image

class VectorBackend(Protocol):
    def embed_region(self, region_bytes: bytes) -> List[float]:
        ...
