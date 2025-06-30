import logging
from typing import Dict, List, Optional, Union

import numpy as np
from database.db_connection import DatabaseConnection
from models.embedding_model import EmbeddingModel
from models.tfidf_vectorizer import TextVectorizer
from preprocessing.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieDatabase:
    def __init__(self, db_connection: DatabaseConnection):
        """
        Инициализация базы данных фильмов.

        Аргументы:
        - `db_connection`: Объект подключения к базе данных.
        """
        self.db_connection = db_connection
        self.embedding_model = EmbeddingModel()
        self.tfidf_model = TextVectorizer()

    def add_movie(self, movie_data: Dict[str, Union[str, float, int]]) -> None:
        """
        Добавляет фильм в базу данных.

        Аргументы:
        - `movie_data`: Данные о фильме в виде словаря.
        """
        # Расчет эмбеддингов
        embedding = self.embedding_model.encode([movie_data["overview"]])[0]
        embedding_json = list(embedding)

        # Расчет TF-IDF-векторов
        if self.tfidf_model.is_fitted:
            tfidf_vector = self.tfidf_model.transform(movie_data["overview"])
        else:
            tfidf_vector = None
        tfidf_vector_json = (
            list(tfidf_vector) if tfidf_vector is not None else None
        )

        query = """
        INSERT INTO movies (
            title, overview, popularity, vote_count, vote_average,
            original_language, genre, poster_url, embedding, tfidf_vector
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            movie_data["title"],
            movie_data["overview"],
            movie_data["popularity"],
            movie_data["vote_count"],
            movie_data["vote_average"],
            movie_data["original_language"],
            movie_data["genre"],
            movie_data["poster_url"],
            embedding_json,
            tfidf_vector_json,
        )
        self.db_connection.execute_query(query, params)

    def search_movies(
        self, query_text: str, method: str = "embedding", top_n: int = 5
    ) -> List[Dict]:
        """
        Выполняет поиск фильмов по текстовому запросу.

        Аргументы:
        - `query_text`: Текст запроса.
        - `method`: Метод поиска ('embedding' или 'tfidf').
        - `top_n`: Количество возвращаемых результатов.

        Возвращает:
        - Список найденных фильмов.
        """
        cleaned_query = preprocess_text(query_text)
        if method == "embedding":
            return self.search_by_embedding(cleaned_query, top_n)
        elif method == "tfidf":
            return self.search_by_tfidf(cleaned_query, top_n)
        else:
            raise ValueError(
                "Invalid search method! Use 'embedding' or 'tfidf'."
            )

    def search_by_tfidf(self, query_text: str, top_n: int = 5) -> List[Dict]:
        """
        Выполняет поиск фильмов по TF-IDF-вектору.

        Аргументы:
        - `query_text`: Текст запроса.
        - `top_n`: Количество возвращаемых результатов.

        Возвращает:
        - Список найденных фильмов.
        """
        # Рассчитываем TF-IDF-вектор для поиска
        query_vector = self.tfidf_model.transform(query_text)
        query_vector = np.array(query_vector)

        # Выбираем фильмы
        query = "SELECT id, title, overview, tfidf_vector, popularity, vote_average, genre, original_language, poster_url FROM movies WHERE tfidf_vector IS NOT NULL;"
        movies = self.db_connection.execute_query(query)
        for movie in movies[:10]:
            logger.info(f"Result: {movie}")
        results = []

        for movie in movies:
            stored_vector = np.array(movie["tfidf_vector"])
            # Рассчитываем схожесть (по косинусной мере)
            similarity = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
            )
            results.append((similarity, movie))

        # Сортируем по схожести
        results = sorted(results, key=lambda x: x[0], reverse=True)

        final_results = []
        for similarity, movie in results[:top_n]:
            # Убираем tfidf_vector из результата
            movie_without_tfidf = {
                key: value
                for key, value in movie.items()
                if key != "tfidf_vector"
            }
            final_results.append(movie_without_tfidf)
            logger.info(f"Query: {query_text}, Result: {movie_without_tfidf})")

        return final_results

    def search_by_embedding(
        self, query_text: str, top_n: int = 5
    ) -> List[Dict]:
        """
        Выполняет поиск фильмов по эмбеддингу.

        Аргументы:
        - `query_text`: Текст запроса.
        - `top_n`: Количество возвращаемых результатов.

        Возвращает:
        - Список найденных фильмов.
        """
        # Кодируем текст запроса в эмбеддинг
        query_embedding = self.embedding_model.encode([query_text])[0]

        # Выбираем фильмы, у которых есть эмбеддинги
        query = "SELECT id, title, overview, embedding, popularity, vote_average, genre, original_language, poster_url FROM movies WHERE embedding IS NOT NULL;"
        movies = self.db_connection.execute_query(query)
        for movie in movies[:10]:
            logger.info(f"Result: {movie}")
        results = []

        for movie in movies:
            stored_embedding = np.array(movie["embedding"])
            # Рассчитываем косинусное сходство
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(stored_embedding)
            )
            results.append((similarity, movie))

        # Сортируем результаты по схожести
        results = sorted(results, key=lambda x: x[0], reverse=True)

        final_results = []
        for similarity, movie in results[:top_n]:
            # Убираем embedding из результата
            movie_without_embedding = {
                key: value for key, value in movie.items() if key != "embedding"
            }
            final_results.append(movie_without_embedding)
            logger.info(
                f"Query: {query_text}, Result: {movie_without_embedding})"
            )

        return final_results

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """
        Получает информацию о фильме по его ID.

        Аргументы:
        - `movie_id`: ID фильма.

        Возвращает:
        - Данные о фильме или None, если фильм не найден.
        """
        query = "SELECT * FROM movies WHERE id = %s;"
        movie = self.db_connection.execute_query(query, (movie_id,))
        if movie:
            return movie[
                0
            ]  # предположим, что ID уникален и вернет только одну запись
        else:
            return None

    def get_top_movies(
        self, sort_by: str = "popularity", limit: int = 10
    ) -> List[Dict]:
        """
        Возвращает топ фильмов по указанному критерию сортировки.

        Аргументы:
        - `sort_by`: Критерий сортировки (например, 'popularity').
        - `limit`: Количество возвращаемых фильмов.

        Возвращает:
        - Список топ-фильмов.
        """
        query = f"""
        SELECT id, title, overview, popularity, vote_average, genre, original_language, poster_url
        FROM movies
        ORDER BY {sort_by} DESC
        LIMIT %s;
        """
        movies = self.db_connection.execute_query(query, (limit,))
        return movies

    def get_top_movies_by_language(
        self, language: str, sort_by: str = "popularity", limit: int = 10
    ) -> List[Dict]:
        """
        Возвращает топ фильмов по указанному языку и критерию сортировки.

        Аргументы:
        - `language`: Код языка для фильтрации фильмов.
        - `sort_by`: Критерий сортировки ('popularity' или 'vote_average').
        - `limit`: Количество фильмов для возврата.

        Возвращает:
        - Список топ-фильмов на указанном языке.
        """
        query = f"""
        SELECT id, title, overview, popularity, vote_average, genre, original_language, poster_url
        FROM movies
        WHERE original_language = %s
        ORDER BY {sort_by} DESC
        LIMIT %s;
        """
        movies = self.db_connection.execute_query(query, (language, limit))
        return movies
