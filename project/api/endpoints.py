import logging
import time

from database.db_connection import DatabaseConnection
from fastapi import APIRouter, HTTPException, Query
from api.models import IndexType, Movie, SearchResponse, SortBy
from search.search_engine import MovieDatabase

# Настройка логирования
logger = logging.getLogger(__name__)

# Инициализация соединения с базой данных
try:
    db = DatabaseConnection(
        db_name="movies_db", user="user", password="password", host="db"
    )
    db.connect()
    logger.info("Connected to the database.")
except Exception as e:
    logger.error(f"Failed to connect to the database: {e}")
    raise HTTPException(status_code=500, detail="Database connection failed")

# Инициализация MovieDatabase
movie_db = MovieDatabase(db)

# Создаем объект маршрутизации FastAPI
router = APIRouter()


@router.get("/search/", response_model=SearchResponse)
async def search_movies(
    query: str = Query(..., description="Текстовый запрос"),
    index_type: IndexType = Query(
        IndexType.embedding, description="Тип индекса: 'tfidf' или 'embedding'"
    ),
    top_n: int = Query(5, description="Количество результатов (топ-N)"),
) -> SearchResponse:
    """
    Выполняет поиск фильмов по текстовому запросу.

    Аргументы:
    - `query`: текст запроса для поиска фильмов.
    - `index_type`: использовать 'embedding' или 'tfidf' для индексации.
    - `top_n`: количество возвращаемых результатов.

    Возвращает:
    - `SearchResponse`: объект, содержащий список найденных фильмов и время выполнения поиска.
    """
    logger.info(
        f"Search requested: query={query}, index_type={index_type}, top_n={top_n}"
    )
    try:
        start_time = time.time()
        results = movie_db.search_movies(
            query_text=query, method=index_type, top_n=top_n
        )
        search_time = time.time() - start_time
        logger.info(
            f"Search completed successfully in {search_time:.2f} seconds."
        )

        return SearchResponse(movies=results, search_time=search_time)
    except ValueError as e:
        logger.error(f"Invalid parameters for search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/movie/{movie_id}")
async def get_movie(movie_id: int) -> Movie:
    """
    Возвращает информацию о фильме по его идентификатору.

    Аргументы:
    - `movie_id`: идентификатор фильма.

    Возвращает:
    - `Movie`: объект фильма.

    Исключения:
    - `HTTPException`: если фильм не найден.
    """
    movie = movie_db.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@router.get("/top_movies/")
def get_top_movies(sort_by: SortBy = SortBy.popularity) -> dict:
    """
    Возвращает топ фильмов, отсортированных по заданному критерию.

    Аргументы:
    - `sort_by`: критерий сортировки ('popularity' или 'vote_average').

    Возвращает:
    - Словарь с ключом 'movies', содержащий список фильмов.

    Исключения:
    - Возвращает ошибку, если критерий сортировки недопустим.
    """
    return {"movies": movie_db.get_top_movies(sort_by.value)}


@router.get("/top_movies_by_language/")
async def get_top_movies_by_language(
    language: str = Query(
        ..., description="Language code for filtering movies"
    ),
    sort_by: SortBy = Query(
        SortBy.popularity,
        description="Sorting criteria: popularity or vote_average",
    ),
    limit: int = Query(10, description="Number of movies to retrieve"),
) -> dict:
    """
    Возвращает топ фильмов по указанному языку и критерию сортировки.

    Аргументы:
    - `language`: код языка для фильтрации фильмов.
    - `sort_by`: критерий сортировки ('popularity' или 'vote_average').
    - `limit`: количество фильмов для возврата.

    Возвращает:
    - Словарь с ключом 'movies', содержащий список фильмов.

    Исключения:
    - `HTTPException`: если фильмы не найдены или произошла ошибка сервера.
    """
    try:
        movies = movie_db.get_top_movies_by_language(
            language, sort_by.value, limit
        )
        if not movies:
            raise HTTPException(
                status_code=404, detail="No movies found for this language."
            )
        return {"movies": movies}
    except Exception as e:
        logger.error(f"Error retrieving top movies by language: {e}")
        raise HTTPException(status_code=500, detail=str(e))
