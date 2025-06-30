import json

import numpy as np

# Подключение к базе данных
from db_connection import DatabaseConnection
from scipy.sparse import load_npz

from datasets import load_dataset

# Загрузка эмбеддингов
with open("../data/embeddings.json", "r") as f:
    embeddings_data = json.load(f)

# Загрузка TF-IDF данных
tfidf_matrix = load_npz("../data/tfidf_matrix.npz")
with open("../data/tfidf_vocabulary.json", "r") as f:
    tfidf_vocabulary = json.load(f)
ids = np.load("../data/tfidf_ids.npy")

# Загрузка датасета
dataset = load_dataset("Pablinho/movies-dataset")
df = dataset["train"].to_pandas()[:1100]

# Проверка наличия столбца 'id'
if "id" not in df.columns:
    # Создаем уникальные идентификаторы, если их нет
    df["id"] = range(len(df))

# Удаляем строки с отсутствующими значениями в столбцах "id" и "Overview"
df = df.dropna(subset=["id", "Overview"]).reset_index(drop=True)
print(df.head(12))
print("Подготовка данных для вставки")

# Подготовка данных для вставки в БД
insertion_data = []
for i, row in df.iterrows():
    record_id = str(row["id"])
    embedding = json.dumps(embeddings_data[record_id])
    tfidf_vector = tfidf_matrix[i].toarray().tolist()[0]
    tfidf_vector_json = json.dumps(tfidf_vector)

    insertion_data.append(
        (
            row["id"],  # id
            row.get("Title", ""),  # title
            row.get("Overview", ""),  # overview
            row.get("Popularity", 0.0),  # popularity
            row.get("Vote_Count", 0),  # vote_count
            row.get("Vote_Average", 0.0),  # vote_average
            row.get("Original_Language", ""),  # original_language
            row.get("Genre", ""),  # genre
            row.get("Poster_Url", ""),  # poster_url
            embedding,  # embedding
            tfidf_vector_json,  # tfidf_vector
        )
    )

# Вставка данных в таблицу movies
insert_query = """
INSERT INTO movies (
    id, title, overview, popularity, vote_count, vote_average,
    original_language, genre, poster_url, embedding, tfidf_vector
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Установление соединения с базой данных
db = DatabaseConnection(
    db_name="movies_db", user="user", password="password", host="localhost"
)
db.connect()

print("Вставка данных в базу")
# Выполнение вставки данных
with db.connection.cursor() as cursor:
    cursor.executemany(insert_query, insertion_data)
    db.connection.commit()

print("Данные успешно вставлены!")
