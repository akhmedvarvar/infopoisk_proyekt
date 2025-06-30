from database.db_connection import DatabaseConnection

class MovieDatabase:
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection

    def add_movie(self, movie_data):
        """Добавление нового фильма в базу данных"""
        query = """
        INSERT INTO movies (title, overview, popularity, vote_count, vote_average, original_language, genre, poster_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            movie_data['title'],
            movie_data['overview'],
            movie_data['popularity'],
            movie_data['vote_count'],
            movie_data['vote_average'],
            movie_data['original_language'],
            movie_data['genre'],
            movie_data['poster_url']
        )
        self.db_connection.execute_query(query, params)

    def update_movie(self, movie_id, movie_data):
        """Обновление информации о фильме"""
        query = """
        UPDATE movies
        SET title = %s, overview = %s, popularity = %s, vote_count = %s, vote_average = %s, original_language = %s, genre = %s, poster_url = %s
        WHERE id = %s
        """
        params = (
            movie_data['title'],
            movie_data['overview'],
            movie_data['popularity'],
            movie_data['vote_count'],
            movie_data['vote_average'],
            movie_data['original_language'],
            movie_data['genre'],
            movie_data['poster_url'],
            movie_id
        )
        self.db_connection.execute_query(query, params)

    def delete_movie(self, movie_id):
        """Удаление фильма из базы данных"""
        query = "DELETE FROM movies WHERE id = %s"
        params = (movie_id,)
        self.db_connection.execute_query(query, params)

    def get_movie_by_id(self, movie_id):
        """Получение фильма по ID"""
        query = "SELECT * FROM movies WHERE id = %s"
        params = (movie_id,)
        return self.db_connection.execute_query(query, params)

    def get_all_movies(self):
        """Получение всех фильмов"""
        query = "SELECT * FROM movies"
        return self.db_connection.execute_query(query)

# Пример использования
if __name__ == "__main__":
    # Подключаемся к базе данных
    db = DatabaseConnection(db_name="movies_db", user="your_username", password="your_password")
    db.connect()

    # Инициализируем объект для работы с фильмами
    movie_db = MovieDatabase(db)

    # Добавляем новый фильм
    new_movie = {
        'title': 'Inception',
        'overview': 'A skilled thief is given a chance at redemption if he can successfully perform an inception.',
        'popularity': 85.5,
        'vote_count': 10234,
        'vote_average': 8.8,
        'original_language': 'en',
        'genre': 'Action, Science Fiction',
        'poster_url': 'https://linktoimage.com'
    }
    movie_db.add_movie(new_movie)

    movies = movie_db.get_all_movies()
    print("All Movies:", movies)

    db.close()
