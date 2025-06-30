from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


# Определяем перечисление для типов индексов
class IndexType(str, Enum):
    tfidf = "tfidf"
    embedding = "embedding"


# Определяем перечисление для критериев сортировки
class SortBy(str, Enum):
    popularity = "popularity"
    vote_average = "vote_average"


# Определяем модель данных для фильма
class Movie(BaseModel):
    title: str
    overview: str
    popularity: Optional[float] = None
    vote_average: Optional[float] = None
    original_language: Optional[str] = None
    genre: Optional[str] = None
    poster_url: Optional[str] = None


# Определяем модель данных для ответа на поиск
class SearchResponse(BaseModel):
    movies: List[Movie]
    search_time: float


# Определяем заготовки для создания новой записи
class MovieCreate(Movie):
    pass


# Определяем заготовки для обновления информации о фильме
class MovieUpdate(Movie):
    pass
