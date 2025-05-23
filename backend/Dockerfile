FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директории для данных моделей
RUN mkdir -p /app/data

# Добавление пути к моделям в PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/data"

# Экспозиция порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main_final:app", "--host", "0.0.0.0", "--port", "8000"]
