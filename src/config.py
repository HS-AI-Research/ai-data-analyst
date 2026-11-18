
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

OUTPUT_LANGUAGE = os.getenv(
    "OUTPUT_LANGUAGE",
    "auto"
)

TEMPERATURE = 0.2
