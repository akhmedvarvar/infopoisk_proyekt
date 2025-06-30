import json
import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка стоп-слов для русского языка
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))


def preprocess_text(text, remove_stopwords=True):
    """
    Очистка и препроцессинг текста.
    :param text: Исходный текст для обработки.
    :param remove_stopwords: Удалять ли стоп-слова.
    :return: Препроцессированный текст.
    """
    text = text.lower()
    
    # Удаление ссылок, эмодзи и HTML-тегов
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Удаление URL
    text = re.sub(r"<.*?>", "", text)                    # Удаление HTML
    text = re.sub(r"[^\x00-\x7Fа-яА-ЯёЁ0-9\s]", "", text)  # Удаление эмодзи и нестандартных символов
    
    # Удаление пунктуации
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Удаление чисел
    text = re.sub(r"\d+", "", text)
    
    # Токенизация: разделяем текст на слова
    tokens = text.split()

    # Удаление стоп-слов
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    # Сборка текста обратно в строку
    clean_text = " ".join(tokens)
    return clean_text

# === Шаг 1. Загрузка данных ===
def load_df(text_column: str):
    """
    Загружает датасет из CSV-файла.
    :param input_file: Путь к CSV-файлу с данными.
    :param text_column: Название колонки, содержащей текстовые данные.
    :return: DataFrame с текстами и идентификаторами.
    """
    print(f"Loading dataset")
    dataset = load_dataset("Pablinho/movies-dataset")
    df = dataset['train'].to_pandas()
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset!")
    

    if "id" not in df.columns:
        # Если в таблице нет id, создадим свои уникальные идентификаторы
        df["id"] = range(len(df))
    
    df = df[["id", text_column]].dropna().reset_index(drop=True)
    df[text_column] = df[text_column].apply(preprocess_text)
    print(f"Loaded {len(df)} rows with text data.")
    return df


# === Шаг 2. Вычисление эмбеддингов ===
def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Вычисляет sentence embeddings с использованием модели SentenceTransformer.
    :param texts: Список текстов.
    :param model_name: Название модели SentenceTransformer для генерации эмбеддингов.
    :return: np.ndarray с эмбеддингами.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Calculating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"Generated embeddings for {len(texts)} texts.")
    return embeddings

# === Шаг 3. Вычисление TF-IDF матрицы ===
def compute_tfidf_matrix(texts, max_features=5000):
    """
    Вычисляет TF-IDF матрицу для текстов.
    :param texts: Список текстов.
    :param max_features: Максимальное число признаков TF-IDF.
    :return: TF-IDF матрица (scipy.sparse.csr_matrix), словарь {word: index}.
    """
    print("Calculating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer.vocabulary_


# === Шаг 4. Сохранение результатов ===
def save_embeddings(embeddings, ids, output_file):
    """
    Сохраняет эмбеддинги в JSON-файл.
    :param embeddings: ndarray с эмбеддингами.
    :param ids: Список идентификаторов текстов.
    :param output_file: Имя файла для сохранения.
    """
    print(f"Saving embeddings to {output_file}...")
    data = {str(id_): emb.tolist() for id_, emb in zip(ids, embeddings)}
    with open(output_file, "w") as f:
        json.dump(data, f)
    print("Embeddings saved successfully!")


def save_tfidf_matrix(tfidf_matrix, ids, vocabulary, output_file_prefix):
    """
    Сохраняет TF-IDF матрицу и словарь признаков.
    :param tfidf_matrix: TF-IDF матрица.
    :param ids: Список идентификаторов текстов.
    :param vocabulary: Словарь {word: index}.
    :param output_file_prefix: Префикс для имени файлов.
    """
    # Преобразование значений словаря vocabulary из numpy.int64 в int
    vocabulary = {word: int(index) for word, index in vocabulary.items()}

    # Сохранение матрицы
    matrix_file = f"{output_file_prefix}_matrix.npz"
    print(f"Saving TF-IDF matrix to {matrix_file}...")
    from scipy.sparse import save_npz
    save_npz(matrix_file, tfidf_matrix)
    
    # Сохранение словаря
    vocab_file = f"{output_file_prefix}_vocabulary.json"
    print(f"Saving TF-IDF vocabulary to {vocab_file}...")
    with open(vocab_file, "w") as f:
        json.dump(vocabulary, f)

    # Сохранение индексов
    index_file = f"{output_file_prefix}_ids.npy"
    print(f"Saving document IDs to {index_file}...")
    np.save(index_file, ids)
    
    print("TF-IDF data saved successfully!")



# === Шаг 5. Основной скрипт ===
if __name__ == "__main__":
    text_column = "Overview"  # Колонка с текстами
    embedding_output_file = "embeddings.json"  # Выходной файл с эмбеддингами
    tfidf_output_prefix = "tfidf"  # Префикс для выхода TF-IDF (без расширений)
    
    # Загрузка данных
    dataset = load_df(text_column)
    texts = dataset[text_column].tolist()
    ids = dataset["id"].tolist()

    # Расчет эмбеддингов
    embeddings = compute_embeddings(texts)

    # Расчет TF-IDF матрицы
    tfidf_matrix, vocabulary = compute_tfidf_matrix(texts)

    # Сохранение результатов
    save_embeddings(embeddings, ids, embedding_output_file)
    save_tfidf_matrix(tfidf_matrix, ids, vocabulary, tfidf_output_prefix)

    print("Processing completed successfully!")
