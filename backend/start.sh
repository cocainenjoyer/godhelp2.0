#!/bin/bash

# Скрипт для запуска рекомендательной системы сериалов

# Проверка наличия необходимых переменных окружения
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
    echo "Настройка переменных окружения..."
    export POSTGRES_USER=postgres
    export POSTGRES_PASSWORD=postgres
    export POSTGRES_HOST=localhost
    export POSTGRES_PORT=5432
    export POSTGRES_DB=movie_recommender
    export SECRET_KEY=your-secret-key
    export TMDB_API_KEY=your-tmdb-api-key
    export POSTER_STORAGE_PATH=./posters
fi

# Создание директории для постеров, если она не существует
if [ ! -d "$POSTER_STORAGE_PATH" ]; then
    echo "Создание директории для постеров..."
    mkdir -p $POSTER_STORAGE_PATH
fi

# Проверка наличия виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python -m venv venv
fi

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate

# Установка зависимостей
echo "Установка зависимостей..."
pip install -r requirements.txt

# Запуск сервера
echo "Запуск сервера..."
uvicorn main_final:app --reload --host 0.0.0.0 --port 8000
