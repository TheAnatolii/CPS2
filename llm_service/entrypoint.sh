#!/bin/bash
# Запускаем Ollama в фоне
ollama serve &
# Ждем инициализации
sleep 5
# Скачиваем модель
ollama pull qwen2.5:0.5b
# Запускаем FastAPI-обертку
uvicorn main:app --host 0.0.0.0 --port 8000
