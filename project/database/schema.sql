-- Если создаем таблицу с нуля:
CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title TEXT,
    overview TEXT,
    popularity FLOAT,
    vote_count INT,
    vote_average FLOAT,
    original_language TEXT,
    genre TEXT,
    poster_url TEXT,
    embedding JSONB,      
    tfidf_vector JSONB  
);

-- Оптимизация запросов: создаем индексы по вхождению данных
CREATE INDEX idx_embedding ON movies USING gin (embedding);
CREATE INDEX idx_tfidf_vector ON movies USING gin (tfidf_vector);
