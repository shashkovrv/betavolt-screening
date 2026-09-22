# Контейнеризация проекта для воспроизводимости (Раздел 11 плана)
FROM python:3.11-slim

WORKDIR /app

# Установка системных утилит
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода и базы данных
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда запуска веб-интерфейса
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]