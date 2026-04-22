from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_response(request: PromptRequest):
    """
    Принимает запрос по HTTP, пересылает его в локальный инстанс Ollama
    с моделью Qwen2.5:0.5b и возвращает ответ.

    Args:
        request (PromptRequest): Объект запроса, содержащий строку prompt.

    Returns:
        dict: JSON-ответ от Ollama LLM.
    """
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:0.5b",
        "prompt": request.prompt,
        "stream": False
    })
    return response.json()
