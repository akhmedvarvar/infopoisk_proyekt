import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict


class DatabaseConnection:
    def __init__(self, db_name: str, user: str, password: str, host: str = 'db', port: int = 5432):
        """
        Инициализирует подключение к базе данных.

        Аргументы:
        - `db_name`: Имя базы данных.
        - `user`: Имя пользователя базы данных.
        - `password`: Пароль пользователя базы данных.
        - `host`: Адрес хоста базы данных.
        - `port`: Номер порта базы данных.
        """
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self) -> None:
        """
        Устанавливает соединение с базой данных.

        Исключения:
        - `Exception`: если не удается подключиться к базе данных.
        """
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                cursor_factory=RealDictCursor
            )
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def close(self) -> None:
        """
        Закрывает соединение с базой данных.
        """
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[Dict]]:
        """
        Выполняет SQL-запрос и возвращает результаты (если применимо).

        Аргументы:
        - `query`: SQL-запрос для выполнения.
        - `params`: Параметры для SQL-запроса.

        Возвращает:
        - Список результатов запроса (если применимо).

        Исключения:
        - `Exception`: если не удается выполнить запрос.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:  # Проверяем, возвращает ли запрос данные
                    return cursor.fetchall()
                self.connection.commit()
        except Exception as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()
            raise

