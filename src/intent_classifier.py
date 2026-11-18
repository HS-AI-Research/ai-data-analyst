
"""
LLM-Based Intent Classifier
===========================

Responsibilities:
- Classify natural-language analytical requests.
- Support multilingual questions.
- Distinguish exploratory analysis from specific queries.
- Return validated, structured JSON-compatible results.
- Avoid exposing dataset records or sensitive values to the LLM.
- Provide a conservative fallback when Ollama is unavailable.

The classifier does not execute queries or calculate metrics.
"""

import json
import re
from typing import Any, Dict, Optional

import pandas as pd

from src.ollama_client import ask_llm


# =========================================================
# CONFIGURATION
# =========================================================

VALID_INTENTS = {
    "exploratory_analysis",
    "analytical_query",
    "unsupported",
}

MAX_COLUMNS_IN_PROMPT = 80
MAX_QUESTION_LENGTH = 4000


# =========================================================
# DATASET METADATA
# =========================================================

def build_dataset_metadata(
    df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Build a small, privacy-conscious dataset description.

    Only column names and data types are included.
    No row values or samples are sent to the model.
    """

    if not isinstance(df, pd.DataFrame):
        return {
            "available": False,
            "columns": [],
        }

    columns = []

    for column in df.columns[:MAX_COLUMNS_IN_PROMPT]:
        series = df[column]

        columns.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "semantic_hint": (
                "datetime"
                if pd.api.types.is_datetime64_any_dtype(series)
                else "numeric"
                if pd.api.types.is_numeric_dtype(series)
                else "boolean"
                if pd.api.types.is_bool_dtype(series)
                else "text_or_category"
            ),
        })

    return {
        "available": True,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "columns_truncated": (
            len(df.columns) > MAX_COLUMNS_IN_PROMPT
        ),
    }


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json_object(response: str) -> Dict[str, Any]:
    """
    Extract a JSON object from an LLM response.

    Supports plain JSON and Markdown fenced JSON.
    """

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty classifier response.")

    text = response.strip()

    # Remove Markdown code fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    # First try parsing the complete response.
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Recover the first JSON object if extra text exists.
    start = text.find("{")

    if start < 0:
        raise ValueError("No JSON object found.")

    decoder = json.JSONDecoder()

    try:
        result, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON classifier response."
        ) from error

    if not isinstance(result, dict):
        raise ValueError("Classifier output must be a JSON object.")

    return result


# =========================================================
# OUTPUT VALIDATION
# =========================================================

def validate_classification(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize classifier output.

    Unknown intent values are rejected.
    """

    if not isinstance(result, dict):
        raise TypeError("Classification must be a dictionary.")

    intent = str(
        result.get("intent", "")
    ).strip().lower()

    if intent not in VALID_INTENTS:
        raise ValueError(
            f"Unsupported intent: {intent!r}"
        )

    confidence = result.get("confidence", 0.0)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if not 0.0 <= confidence <= 1.0:
        confidence = 0.0

    scope = result.get("analysis_scope", [])

    if not isinstance(scope, list):
        scope = []

    # Scope is descriptive metadata, not executable instructions.
    scope = [
        str(item).strip()[:80]
        for item in scope[:20]
        if isinstance(item, str) and item.strip()
    ]

    return {
        "intent": intent,
        "confidence": confidence,
        "analysis_scope": scope,
        "reason": str(result.get("reason", ""))[:500],
    }



def _is_specific_analytical_request(question: str) -> bool:
    """
    Detect explicit calculations and aggregations before using the LLM.
    This prevents specific analytical questions from being routed to EDA.
    """
    if not isinstance(question, str) or not question.strip():
        return False

    text = question.lower().strip()

    text = (
        text.replace("\u064a", "\u06cc")
            .replace("\u0643", "\u06a9")
            .replace("\u200c", "")
            .replace("\u200e", "")
            .replace("\u200f", "")
    )

    compact = re.sub(r"[\s\u0600-\u06ff]+", lambda m: re.sub(r"\s+", "", m.group(0)), text)

    # Persian: distinct-value counting.
    count_word = "\u062a\u0639\u062f\u0627\u062f"
    distinct_word = "\u0645\u062a\u0641\u0627\u0648\u062a"
    split_word = "\u0628\u0647\u062a\u0641\u06a9\u06cc\u06a9"

    if count_word in compact and distinct_word in compact:
        return True

    # Persian: explicit metric grouped by a dimension.
    metric_words = (
        count_word,
        "\u0645\u062c\u0645\u0648\u0639",
        "\u0645\u06cc\u0627\u0646\u06af\u06cc\u0646",
        "\u0645\u06cc\u0627\u0646\u06af\u06cc\u0646\u0647",
        "\u062d\u062f\u0627\u06a9\u062b\u0631",
        "\u062d\u062f\u0627\u0642\u0644",
    )

    if split_word in compact and any(word in compact for word in metric_words):
        return True

    # English: explicit distinct counts.
    english_patterns = [
        r"\b(count|number)\s+of\s+distinct\b",
        r"\bdistinct\s+(count|number)\b",
        r"\bhow\s+many\s+distinct\b",
        r"\bcount\s+distinct\b",
        r"\bgroup\s+by\b",
        r"\b(count|sum|average|mean|median|min|max|total)\b.*\bby\b",
    ]

    if any(re.search(pattern, text) for pattern in english_patterns):
        return True

    return False


# =========================================================
# FALLBACK CLASSIFIER
# =========================================================

def fallback_classification(
    question: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Conservative fallback when the LLM is unavailable.

    The existing EDA detector is retained only as a fallback.
    It is no longer the primary multilingual classifier.
    """

    if _is_specific_analytical_request(question):
        return {
            "intent": "analytical_query",
            "confidence": 0.99,
            "analysis_scope": [],
            "reason": "Explicit analytical operation detected by rules.",
            "classifier": "rule_based",
            "error": str(error) if error else None,
        }

    intent = "analytical_query"

    try:
        from src.eda_engine import is_exploratory_request

        if is_exploratory_request(question):
            intent = "exploratory_analysis"

    except Exception:
        pass

    return {
        "intent": intent,
        "confidence": 0.0,
        "analysis_scope": [],
        "reason": "Fallback classification.",
        "classifier": "fallback",
        "error": str(error) if error else None,
    }


# =========================================================
# CLASSIFIER PROMPT
# =========================================================

def build_classifier_prompt(
    question: str,
    dataset_metadata: Dict[str, Any],
) -> str:
    """Build a constrained multilingual classification prompt."""

    metadata_json = json.dumps(
        dataset_metadata,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are a multilingual intent classifier for a general-purpose
data analysis application.

Understand the user's question in its original language.
Do not translate the question unnecessarily.

Your task is to classify the user's analytical intent.

Allowed intent values:

1. exploratory_analysis
   Use when the user asks for broad investigation of a dataset,
   business insights, risks, patterns, anomalies, data quality,
   general findings, or an overall exploratory analysis.

2. analytical_query
   Use when the user requests a specific calculation, metric,
   aggregation, comparison, ranking, grouping, filter, or trend.

3. unsupported
   Use only when the request is unrelated to data analysis
   or cannot reasonably be interpreted as a data-analysis request.

Important distinctions:

- "What are the most important business insights and risks?"
  => exploratory_analysis

- "Which products have the most orders?"
  => analytical_query

- "Show total revenue by city."
  => analytical_query

- "Find missing values and unusual patterns."
  => exploratory_analysis

- A request for one specific statistic is not exploratory analysis.
- Do not classify a request as exploratory merely because it
  contains words equivalent to "most" or "important".
- Use the meaning of the complete sentence, not isolated keywords.
- Understand Persian, English, and other languages.
- Do not execute calculations.
- Do not invent column names.
- Dataset metadata is provided only as context.

Return ONLY a valid JSON object with this structure:

{{
  "intent": "exploratory_analysis",
  "confidence": 0.95,
  "analysis_scope": [
    "data_quality",
    "business_insights",
    "potential_risks"
  ],
  "reason": "Brief explanation"
}}

USER QUESTION:
{question[:MAX_QUESTION_LENGTH]}

DATASET METADATA:
{metadata_json}
"""


# =========================================================
# MAIN CLASSIFICATION
# =========================================================

def classify_intent(
    question: str,
    df: Optional[pd.DataFrame] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a user request using an explicitly selected Ollama model.

    Args:
        question: Natural-language user request.
        df: Optional dataset.
        model:
            - Model name: use that model.
            - None: use fallback classification.

    Returns:
        A validated classification dictionary.

    This function never executes analytical operations.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    question = question.strip()

    if _is_specific_analytical_request(question):
        return {
            "intent": "analytical_query",
            "confidence": 0.99,
            "analysis_scope": [],
            "reason": "Explicit analytical operation detected by rules.",
            "classifier": "rule_based",
            "error": None,
        }

    if model is None:
        return fallback_classification(question)

    metadata = build_dataset_metadata(df)

    prompt = build_classifier_prompt(
        question,
        metadata,
    )

    try:
        response = ask_llm(
            prompt=prompt,
            model=model,
        )

        raw_result = extract_json_object(response)

        classification = validate_classification(
            raw_result
        )

        classification["classifier"] = "llm"
        classification["error"] = None

        return classification

    except Exception as error:
        return fallback_classification(
            question,
            error=str(error),
        )
