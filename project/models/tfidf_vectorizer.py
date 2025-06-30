import pickle
import numpy as np
from scipy.sparse import csr_matrix
from typing import List

class TextVectorizer:
    def __init__(self) -> None:
        """
        Инициализирует TextVectorizer, загружая векторизатор TF-IDF из файла.
        
        Используются данные из файла `tfidf_vectorizer.pkl`, который должен находиться в папке `data`.
        """
        with open("data/tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def transform(self, text: str) -> np.ndarray:
        """
        Преобразует текстовый документ в векторное представление с использованием TF-IDF.

        Args:
            text (str): Текстовый документ для преобразования.

        Returns:
            np.ndarray: Векторное представление текста в виде плотного массива.
        """
        vector: csr_matrix = self.vectorizer.transform([text])
        dense_vector: np.ndarray = vector.toarray()[0]  # Преобразуем в массив
        return dense_vector

