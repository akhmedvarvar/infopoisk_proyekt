# Подключение к базе данных
from db_connection import DatabaseConnection

# Инициализация подключения к базе данных
db = DatabaseConnection(
    db_name="movies_db",
    user="user",
    password="password",
    host="localhost"
)

try:
    # Установление соединения с базой данных
    db.connect()
    print("Соединение с базой данных установлено.")
    
    # Удаление существующей таблицы movies, если она есть
    drop_table_query = "DROP TABLE IF EXISTS movies;"
    db.execute_query(drop_table_query)
    print("Существующая таблица movies удалена.")
    
    # Чтение и исполнение SQL-схемы для создания новой таблицы
    with open("schema.sql", "r") as f:
        schema = f.read()
        db.execute_query(schema)
    print("Таблица успешно создана.")

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    # Закрытие соединения с базой данных
    db.close()
    print("Соединение с базой данных закрыто.")

