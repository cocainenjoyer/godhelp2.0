FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создание директории для данных моделей и постеров
RUN mkdir -p /app/data
RUN mkdir -p /app/posters

# Копирование моделей и данных
COPY final_model.py /app/data/
COPY similar_prod.py /app/data/
COPY data_final.pkl /app/data/
COPY description_embeddings.npy /app/data/
COPY keywords_embeddings.npy /app/data/

# Добавление пути к моделям в PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/data"
ENV POSTER_STORAGE_PATH="/app/posters"
ENV TMDB_API_KEY="your-tmdb-api-key"

# Экспозиция порта
EXPOSE 8000

# Запуск приложения
CMD ["python", "-c", "import sys; print(sys.path)"]
