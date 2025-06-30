import logging
from typing import Dict

import httpx
import streamlit as st

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Заголовок
st.title("Movie Search Engine")

# Ввод данных
query = st.text_input("Enter your search query:")
method = st.selectbox("Choose search method", ["tfidf", "embedding"])
top_n = st.slider("Number of results", min_value=1, max_value=10, value=5)

# Инициализация состояния сессии, если оно еще не создано
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "top_movies" not in st.session_state:
    st.session_state.top_movies = None

# Кнопка для запуска поиска
if st.button("Search"):
    if query:
        logger.info(
            f"Starting search for query: {query}, method: {method}, top_n: {top_n}"
        )

        try:
            with st.spinner("Searching..."):
                response = httpx.get(
                    f"http://backend:8000/search/",
                    params={
                        "query": query,
                        "index_type": method,
                        "top_n": top_n,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                response_data = response.json()
                st.session_state.search_results = (
                    response_data  # Сохранение результатов в состоянии сессии
                )

        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            st.error("Failed to connect to the server. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error("Something went wrong. Please try again.")
    else:
        st.warning("Please enter a query.")

# Отображение результатов поиска, если они есть
if st.session_state.search_results is not None:
    search_time = st.session_state.search_results.get("search_time", 0)
    movies = st.session_state.search_results.get("movies", [])

    st.info(f"Search completed in {search_time:.2f} seconds.")

    if movies:
        for idx, movie in enumerate(movies):
            st.subheader(f"{idx + 1}. {movie['title']}")
            st.write(f"Overview: {movie['overview']}")
            if movie.get("popularity") is not None:
                st.write(f"Popularity: {movie['popularity']}")
            if movie.get("vote_average") is not None:
                st.write(f"Vote Average: {movie['vote_average']}")
            if movie.get("genre") is not None:
                st.write(f"Genre: {movie['genre']}")
            if movie.get("original_language") is not None:
                st.write(f"Language: {movie['original_language']}")
            st.markdown("---")
    else:
        st.warning("No results found.")

st.header("Recommendations and top movies")

sort_criteria = st.selectbox(
    "Sort top movies by", ["popularity", "vote_average"]
)

if st.button("Show Top 10 Movies"):
    try:
        with st.spinner("Fetching top movies..."):
            response = httpx.get(
                f"http://backend:8000/top_movies/",
                params={"sort_by": sort_criteria},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.top_movies = response.json().get(
                "movies", []
            )  # Сохранение результатов в состоянии сессии

    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}")
        st.error("Failed to connect to the server. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("Something went wrong. Please try again.")

# Отображение топ-10 фильмов, если они есть
if st.session_state.top_movies is not None:
    st.info(f"Showing top 10 movies sorted by {sort_criteria}.")
    for idx, movie in enumerate(st.session_state.top_movies):
        st.subheader(f"{idx + 1}. {movie['title']}")
        st.write(f"Overview: {movie['overview']}")
        st.write(f"Popularity: {movie['popularity']}")
        st.write(f"Vote Average: {movie['vote_average']}")
        st.markdown("---")


def get_language_display_name(language_code: str) -> str:
    """
    Получает отображаемое название языка на основании его кода.

    Args:
        language_code (str): Код языка (например, 'en', 'ja').

    Returns:
        str: Название языка на английском языке, либо исходный код, если язык не найден в списке.
    """
    language_map: Dict[str, str] = {
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "es": "Spanish",
        "ru": "Russian",
        "de": "German",
        "cn": "Chinese",
        "it": "Italian",
    }
    return language_map.get(language_code, language_code)


# Добавляем выбор языка
language_code = st.selectbox(
    "Choose a language",
    ["en", "ja", "fr", "es", "ru", "de", "cn", "it"],
    format_func=get_language_display_name,
)

if st.button("Show Top Movies by Language"):
    try:
        with st.spinner("Fetching top movies by language..."):
            response = httpx.get(
                f"http://backend:8000/top_movies_by_language/",
                params={"language": language_code, "sort_by": sort_criteria},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.top_movies_by_language = response.json().get(
                "movies", []
            )  # Сохранение результатов в состоянии сессии

    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}")
        st.error("Failed to connect to the server. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("Something went wrong. Please try again.")

# Отображение топ фильмов по языку, если они есть
if st.session_state.get("top_movies_by_language") is not None:
    st.info(
        f"Showing top movies in {get_language_display_name(language_code)} sorted by {sort_criteria}."
    )
    for idx, movie in enumerate(st.session_state.top_movies_by_language):
        st.subheader(f"{idx + 1}. {movie['title']}")
        st.write(f"Overview: {movie['overview']}")
        st.write(f"Popularity: {movie['popularity']}")
        st.write(f"Vote Average: {movie['vote_average']}")
        st.write(
            f"Language: {get_language_display_name(movie['original_language'])}"
        )
        st.markdown("---")
