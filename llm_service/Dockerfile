FROM ubuntu:22.04

# Установка зависимостей
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip curl zstd && \
    rm -rf /var/lib/apt/lists/*

# Установка Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

# Запуск скрипта
CMD ["./entrypoint.sh"]
