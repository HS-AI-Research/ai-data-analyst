
"""
Ollama Client Manager
=====================

Responsibilities:
- Manage the Ollama connection.
- Check Ollama availability.
- Retrieve locally installed models.
- Validate model availability.
- Send prompts to a selected model.
- Preserve backward compatibility with ask_llm(prompt).

The application layer is responsible for selecting
the active model and passing its name to this module.
"""

from ollama import Client

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE,
)


# =========================================================
# CLIENT
# =========================================================

client = Client(
    host=OLLAMA_HOST
)


# =========================================================
# RESPONSE UTILITIES
# =========================================================

def get_response_field(response, field, default=None):
    """Read a field from either a dictionary or an SDK object."""

    if isinstance(response, dict):
        return response.get(field, default)

    return getattr(
        response,
        field,
        default
    )


# =========================================================
# CONNECTION STATUS
# =========================================================

def check_ollama_connection():
    """
    Check whether Ollama is reachable.

    Returns:
        {
            "connected": bool,
            "models": list,
            "error": str | None
        }
    """

    try:
        response = client.list()

        models = get_response_field(
            response,
            "models",
            []
        )

        installed_models = []

        for item in models:
            name = get_response_field(
                item,
                "model"
            )

            if name:
                installed_models.append(name)

        return {
            "connected": True,
            "models": sorted(set(installed_models)),
            "error": None,
        }

    except Exception as error:
        return {
            "connected": False,
            "models": [],
            "error": str(error),
        }


# =========================================================
# INSTALLED MODELS
# =========================================================

def get_installed_models():
    """
    Return locally installed Ollama model names.

    Raises:
        ConnectionError: If Ollama is unavailable.
    """

    status = check_ollama_connection()

    if not status["connected"]:
        raise ConnectionError(
            f"Cannot connect to Ollama: {status['error']}"
        )

    return status["models"]


# =========================================================
# MODEL VALIDATION
# =========================================================

def validate_model(model=None):
    """
    Validate the requested model against installed models.

    Returns:
        Selected model name.
    """

    selected_model = model or OLLAMA_MODEL

    installed_models = get_installed_models()

    if selected_model not in installed_models:
        raise ValueError(
            f"Model '{selected_model}' is not installed. "
            f"Installed models: {installed_models}"
        )

    return selected_model


# =========================================================
# LLM REQUEST
# =========================================================

def ask_llm(prompt: str, model=None) -> str:
    """
    Send a prompt to the selected Ollama model.

    Args:
        prompt: User or application prompt.
        model: Optional model name.

    Returns:
        Generated response text.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "Prompt must be a non-empty string."
        )

    selected_model = validate_model(model)

    response = client.chat(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise and reliable multilingual AI assistant. "
                    "Follow the language requested by the user or application prompt. "
                    "Follow the requested output format exactly, including JSON when required. "
                    "Never invent facts, statistics, measurements, or analytical results. "
                    "Use only the information and evidence supplied in the prompt. "
                    "Do not claim to have performed calculations that were not provided."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": TEMPERATURE,
        },
    )

    message = get_response_field(
        response,
        "message",
        {}
    )

    content = get_response_field(
        message,
        "content",
        ""
    )

    if not content:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return content.strip()
