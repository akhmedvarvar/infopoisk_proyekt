from db_connection import DatabaseConnection

# Подключение к базе данных
db = DatabaseConnection(db_name="movies_db", user="user", password="password", host="localhost")

# Установите соединение
db.connect()

try:
    # Создание индексов
    create_index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_embedding ON movies USING gin (embedding);",
        "CREATE INDEX IF NOT EXISTS idx_tfidf_vector ON movies USING gin (tfidf_vector);"
    ]

    with db.connection.cursor() as cursor:
        for query in create_index_queries:
            cursor.execute(query)
        db.connection.commit()

    print("Indexes created successfully!")

finally:
    # Закрытие соединения
    db.close()
