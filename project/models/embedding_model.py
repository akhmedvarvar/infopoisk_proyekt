from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """
        Инициализирует модель для создания эмбеддингов, используя SentenceTransformers.

        Args:
            model_name (str): Название предобученной модели, используемой для создания эмбеддингов.
        """
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Кодирует список текстов в эмбеддинги.

        Args:
            texts (List[str]): Список предварительно обработанных текстовых документов.

        Returns:
            np.ndarray: Массив эмбеддингов для входных текстов.
        """
        return self.model.encode(texts, convert_to_numpy=True)
