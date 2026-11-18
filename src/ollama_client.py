
from ollama import Client
from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE
)

client = Client(host=OLLAMA_HOST)


def get_installed_models():
    """Return all locally installed Ollama models."""
    response = client.list()
    return [model.model for model in response.models]


def ask_llm(prompt: str) -> str:
    """Send a prompt and respond in the same language as the user."""

    installed_models = get_installed_models()

    if OLLAMA_MODEL not in installed_models:
        raise ValueError(
            f"Model '{OLLAMA_MODEL}' not found.\n"
            f"Installed models: {installed_models}"
        )

    system_prompt = """
You are a professional data analyst.

Rules:
1. Detect the language of the user's question.
2. Respond in exactly the same language as the user's question.
3. If the question is in Persian, respond entirely in Persian.
4. If the question is in English, respond entirely in English.
5. Apply the same rule to headings, explanations, and conclusions.
6. Keep numbers, dates, and technical terms accurate.
7. Use evidence-based reasoning.
8. Never invent statistical results.
9. Do not translate column names or raw data values unless necessary.
"""

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": TEMPERATURE
        }
    )

    return response.message.content
