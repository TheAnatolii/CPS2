import os

import faiss
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

mcp = FastMCP("Petfinder_RAG")

embedder = None
index = None
df = None


def init_rag():
    global embedder, index, df
    print("Загрузка Sentence Transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ИСПОЛЬЗОВАНИЕ ДАТАСЕТА: читаем реальный файл
    dataset_path = "dataset.csv"
    print(f"Загрузка реального датасета {dataset_path}...")
    df = pd.read_csv(dataset_path)
    # Оставляем только нужные колонки (Описание и Целевая переменная - Скорость усыновления)
    # В Petfinder колонки называются 'Description' и 'AdoptionSpeed'
    df = df[["Description", "AdoptionSpeed"]].dropna()

    print("Создание FAISS индекса...")
    descriptions = df["Description"].tolist()
    embeddings = embedder.encode(descriptions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    print("Индекс успешно создан. Система RAG готова.")


@mcp.tool()
def retrieve_similar_pets(query: str) -> str:
    """Тулза RAG: находит похожие профили в базе FAISS."""
    if index is None:
        init_rag()
    vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(vec, 1)

    idx = indices[0][0]
    if idx < len(df):
        row = df.iloc[idx]
        return f"Похожее: {row['Description']} | Скорость усыновления: {row['AdoptionSpeed']}"
    return "Нет данных."


@mcp.tool()
def calculate_description_quality(description: str) -> str:
    """
    Анализирует описание и возвращает метку класса для принятия решений клиентом.
    Возможные значения: "SHORT", "AVERAGE", "GOOD".
    """
    words_count = len(description.split())
    if words_count < 4:
        return "SHORT"
    elif words_count < 10:
        return "AVERAGE"
    else:
        return "GOOD"


app = FastAPI()


class QueryRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup_event():
    # Запускаем создание FAISS базы при старте контейнера
    init_rag()


@app.post("/process")
def process_request(req: QueryRequest):
    """
    Умный MCP-клиент: применяет тулы и в зависимости от качества текста
    маршрутизирует задачу (формирует разные промпты для LLM).
    """
    # 1. Применяем инструменты
    rag_context = retrieve_similar_pets(req.prompt)
    quality_category = calculate_description_quality(req.prompt)

    # 2. ДИНАМИЧЕСКИЙ ПРОМПТИНГ (Роутинг на основе тула 2)
    if quality_category == "SHORT":
        # Описание слишком короткое. Просим сгенерировать вопросы для волонтера.
        enriched_prompt = (
            f"Системная информация: В приют поступило животное, но описание слишком короткое: '{req.prompt}'.\n"
            f"Справка из базы RAG: {rag_context}\n\n"
            f"Задача:\n"
            f"1) Не пытайся предсказывать сроки (мало данных).\n"
            f"2) Напиши список из 3 конкретных вопросов для волонтера, чтобы узнать важные детали о животном.\n"
            f"3) Напиши короткий черновик пиар-текста, используя базу RAG для вдохновения."
        )

    elif quality_category == "AVERAGE":
        # Среднее описание. Просим предсказать срок и переписать текст.
        enriched_prompt = (
            f"Системная информация: Поступило животное со стандартным описанием: '{req.prompt}'.\n"
            f"Справка из базы RAG: {rag_context}\n\n"
            f"Задача:\n"
            f"1) Опираясь на RAG, предскажи срок усыновления.\n"
            f"2) Перепиши исходное описание, сделав его более эмоциональным и привлекательным для соцсетей."
        )

    else:  # GOOD
        # Отличное описание. Хвалим, предсказываем и даем советы по пиару.
        enriched_prompt = (
            f"Системная информация: Поступило животное с отличным подробным описанием: '{req.prompt}'.\n"
            f"Справка из базы RAG: {rag_context}\n\n"
            f"Задача:\n"
            f"1) Опираясь на RAG, предскажи срок усыновления.\n"
            f"2) Дай 2 коротких совета по пиару (например, какую фотографию лучше прикрепить или где разместить объявление)."
        )

    # 3. Отправляем финальный умный промпт в LLM
    try:
        llm_response = requests.post(
            "http://llm-service:8000/generate",
            json={"prompt": enriched_prompt},
            timeout=120,
        )
        llm_verdict = llm_response.json().get("response", "Ошибка в ответе LLM")
    except Exception as e:
        llm_verdict = f"Ошибка сети при запросе к LLM: {str(e)}"

    return {
        "user_query": req.prompt,
        "quality_category": quality_category,
        "llm_verdict": llm_verdict,
    }
