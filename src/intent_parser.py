
"""
General-Purpose Intent Parser
Version: 4.0

Responsibilities:
- Interpret Persian and English analytical questions.
- Understand arbitrary CSV and Excel datasets.
- Resolve registered business metrics.
- Detect physical columns and semantic dimensions.
- Infer statistical operations and aggregations.
- Support grouped analysis and ranking.
- Support distinct counts.
- Support filtering and time-series analysis.
- Use Ollama only when deterministic parsing is insufficient.
- Validate queries before execution.

This module does not perform numerical calculations.
All calculations belong to query_engine.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

import pandas as pd

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE,
)

from src.dataset_context import build_dataset_context

from src.metric_registry import (
    METRIC_REGISTRY,
    get_metric,
    resolve_metric,
)

try:
    from ollama import Client
except ImportError:
    Client = None


# =========================================================
# CONFIGURATION
# =========================================================

PARSER_VERSION = "4.0"

ALLOWED_OPERATIONS = {
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "groupby",
    "filter",
    "return_rate",
    "trend_by_date",
}

ALLOWED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "count",
    "nunique",
    "min",
    "max",
}

ALLOWED_FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
}

ALLOWED_GRANULARITIES = {
    "day",
    "week",
    "month",
    "quarter",
    "year",
}

MAX_TOP_K = 50

GENERIC_OPERATION_METRICS = {
    "sum",
    "average",
    "mean",
    "median",
    "minimum",
    "maximum",
    "min",
    "max",
    "count",
}

PERSIAN_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value: Any) -> str:
    """
    Normalize Persian and English text.

    Handles Unicode normalization, Persian characters,
    Arabic characters, digits, ZWNJ and punctuation.
    """

    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(PERSIAN_DIGITS)

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "\u200c": " ",
        "\u200f": " ",
        "\u200e": " ",
        "_": " ",
        "-": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s\u0600-\u06FF.]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(text.split())

def singularize(word: str) -> str:
    """
    Normalize common English and Persian plural forms.
    """

    word = normalize_text(word)

    if not word:
        return word

    # English plural forms
    if re.fullmatch(r"[a-z0-9]+", word):
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"

        if word.endswith("sses") and len(word) > 4:
            return word[:-2]

        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]

        return word

    # Persian plural forms
    if word.endswith("های") and len(word) > 3:
        return word[:-3].strip()

    if word.endswith("ها") and len(word) > 2:
        return word[:-2].strip()

    if word.endswith("ات") and len(word) > 3:
        return word[:-2].strip()

    return word.strip()

def contains_phrase(text: str, phrases) -> bool:
    """
    Check for a phrase.

    English phrases use word boundaries.
    Persian phrases use normalized substring matching.
    """

    normalized = normalize_text(text)

    for phrase in phrases:
        candidate = normalize_text(phrase)

        if not candidate:
            continue

        if re.search(r"[\u0600-\u06FF]", candidate):
            if candidate in normalized:
                return True
        else:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(candidate)
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, normalized):
                return True

    return False


def contains_any_word(text: str, words) -> bool:
    """Check complete words."""

    tokens = set(normalize_text(text).split())

    return any(
        normalize_text(word) in tokens
        for word in words
    )


# =========================================================
# DATASET CONTEXT
# =========================================================

def get_dataset_context(df: pd.DataFrame) -> dict:
    """Build the unified dataset context."""

    return build_dataset_context(df)


def get_schema_map(context: dict) -> dict:
    """Return a column-to-schema mapping."""

    return {
        item["column"]: item
        for item in context.get("schema", [])
        if isinstance(item, dict) and item.get("column")
    }


def get_columns_by_role(context: dict, role: str) -> list[str]:
    """Return columns matching a semantic role."""

    return [
        item["column"]
        for item in context.get("schema", [])
        if (
            isinstance(item, dict)
            and item.get("semantic_role") == role
            and item.get("column")
        )
    ]


def get_numeric_columns(
    df: pd.DataFrame,
    context: Optional[dict] = None,
) -> list[str]:
    """Identify numeric measure columns."""

    if context is not None:
        columns = get_columns_by_role(
            context,
            "numeric_measure",
        )

        if columns:
            return [
                column
                for column in columns
                if column in df.columns
            ]

    return [
        column
        for column in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        )
    ]


def get_categorical_columns(
    df: pd.DataFrame,
    context: Optional[dict] = None,
) -> list[str]:
    """Identify categorical dimensions."""

    if context is not None:
        columns = get_columns_by_role(
            context,
            "categorical_dimension",
        )

        if columns:
            return [
                column
                for column in columns
                if column in df.columns
            ]

    result = []

    for column in df.columns:
        series = df[column]

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            if series.nunique(dropna=True) > 1:
                result.append(column)

    return result


def get_boolean_columns(
    df: pd.DataFrame,
    context: Optional[dict] = None,
) -> list[str]:
    """Identify Boolean-like columns."""

    if context is not None:
        columns = get_columns_by_role(
            context,
            "boolean",
        )

        if columns:
            return [
                column
                for column in columns
                if column in df.columns
            ]

    true_values = {
        "true", "yes", "1", "y", "بله",
    }

    false_values = {
        "false", "no", "0", "n", "خیر",
    }

    result = []

    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            result.append(column)
            continue

        values = {
            normalize_text(value)
            for value in df[column].dropna().unique()
        }

        if values and values.issubset(
            true_values | false_values
        ):
            result.append(column)

    return result


def get_date_columns(
    df: pd.DataFrame,
    context: Optional[dict] = None,
) -> list[str]:
    """Identify date and datetime columns."""

    if context is not None:
        columns = get_columns_by_role(
            context,
            "date",
        )

        if columns:
            return [
                column
                for column in columns
                if column in df.columns
            ]

    result = []

    date_hints = {
        "date",
        "time",
        "timestamp",
        "datetime",
        "تاریخ",
        "زمان",
    }

    for column in df.columns:
        series = df[column]
        normalized = normalize_text(column)

        if pd.api.types.is_datetime64_any_dtype(series):
            result.append(column)
            continue

        if any(
            hint in normalized.split()
            for hint in date_hints
        ):
            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            if parsed.notna().any():
                result.append(column)

    return list(dict.fromkeys(result))


# =========================================================
# COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {
    "quantity": [
        "quantity", "quantities", "units",
        "items sold", "تعداد", "مقدار",
        "اقلام", "تعداد اقلام", "میزان",
    ],
    "price": [
        "price", "unit price", "product price",
        "قیمت", "بهای واحد", "قیمت واحد",
    ],
    "revenue": [
        "revenue", "sales revenue", "income",
        "turnover", "درآمد", "درآمد فروش",
        "مبلغ فروش",
    ],
    "discount": [
        "discount", "discount percentage",
        "تخفیف", "درصد تخفیف",
    ],
    "rating": [
        "rating", "customer rating", "score",
        "امتیاز", "امتیاز مشتری", "نمره",
    ],
    "date": [
        "date", "time", "timeline",
        "تاریخ", "زمان",
    ],
    "city": [
        "city", "cities", "شهر", "شهرها",
    ],
    "product": [
        "product", "products", "item", "items",
        "محصول", "محصولات", "کالا", "کالاها",
    ],
    "category": [
        "category", "categories",
        "دسته", "دسته بندی", "گروه محصول",
    ],
    "customer": [
        "customer", "client", "buyer",
        "مشتری", "خریدار",
    ],
    "employee": [
        "employee", "staff", "worker",
        "کارمند", "کارکنان", "پرسنل",
    ],
    "department": [
        "department", "departments",
        "دپارتمان", "دپارتمان ها",
        "بخش", "واحد سازمانی",
    ],
    "age": [
        "age", "سن",
    ],
    "salary": [
        "salary", "wage", "pay",
        "compensation", "حقوق", "دستمزد",
        "میزان حقوق",
    ],
    "experience": [
        "experience", "work experience",
        "tenure", "سابقه", "سابقه کاری",
        "تجربه کاری",
    ],
    "performance": [
        "performance", "performance score",
        "employee performance", "عملکرد",
        "عملکرد کارکنان", "امتیاز عملکرد",
    ],
    "region": [
        "region", "area", "territory",
        "منطقه", "ناحیه",
    ],
    "payment": [
        "payment", "payment method",
        "روش پرداخت", "پرداخت",
    ],
    "status": [
        "status", "state", "وضعیت", "حالت",
    ],
    "gender": [
        "gender", "sex", "جنسیت",
    ],
}


# =========================================================
# METRIC ALIASES
# =========================================================

METRIC_ALIASES = {
    "revenue": [
        "revenue", "sales revenue", "total sales",
        "فروش", "میزان فروش", "درآمد فروش",
        "مجموع فروش", "ارزش فروش",
    ],
    "quantity": [
        "total quantity", "quantity sold",
        "مجموع تعداد", "تعداد فروش", "مقدار فروش",
    ],
    "orders": [
        "orders", "order count", "number of orders",
        "تعداد سفارش", "تعداد سفارش ها",
        "تعداد سفارشات",
    ],
    "return_rate": [
        "return rate", "returned rate",
        "نرخ مرجوعی", "درصد مرجوعی",
    ],
    "average_price": [
        "average price", "mean price",
        "میانگین قیمت", "متوسط قیمت",
    ],
    "average_rating": [
        "average rating",
        "میانگین امتیاز", "متوسط امتیاز",
    ],
    "discount": [
        "discount", "average discount",
        "تخفیف", "میانگین تخفیف",
    ],
    "employee_count": [
        "employee count", "number of employees",
        "تعداد کارکنان", "تعداد کارمندان",
        "تعداد پرسنل",
    ],
    "total_salary": [
        "total salary", "total salaries",
        "مجموع حقوق", "جمع حقوق", "کل حقوق",
        "مجموع دستمزد",
    ],
    "average_salary": [
        "average salary", "mean salary",
        "میانگین حقوق", "متوسط حقوق",
        "میانگین دستمزد",
    ],
    "average_age": [
        "average age", "mean age",
        "میانگین سن", "متوسط سن",
    ],
    "average_experience": [
        "average experience", "mean experience",
        "میانگین سابقه", "متوسط سابقه",
        "میانگین سابقه کاری",
    ],
    "average_performance": [
        "average performance", "mean performance",
        "میانگین عملکرد", "متوسط عملکرد",
        "میانگین امتیاز عملکرد",
    ],
}


# =========================================================
# COLUMN MATCHING
# =========================================================

def _english_boundary_match(text: str, phrase: str) -> bool:
    """Match an English phrase using word boundaries."""

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return bool(re.search(pattern, text))

def _phrase_match(text: str, phrase: str) -> bool:
    """Match normalized Persian or English phrases."""

    text = normalize_text(text)
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    if re.search(r"[\u0600-\u06FF]", phrase):
        if phrase in text:
            return True

        text_words = text.split()
        phrase_words = phrase.split()

        if len(text_words) != len(phrase_words):
            return False

        return all(
            singularize(text_word) == singularize(phrase_word)
            for text_word, phrase_word in zip(
                text_words,
                phrase_words,
            )
        )

    return _english_boundary_match(text, phrase)

def find_column_in_question(question, columns):
    """
    Find a physical column explicitly mentioned in a question.

    Longest matching name wins.
    Ambiguous equal-length matches return None.
    """

    q = normalize_text(question)
    matches = []

    for column in columns:
        normalized = normalize_text(column)

        if not normalized:
            continue

        if _phrase_match(q, normalized):
            matches.append((len(normalized), column))
            continue

        if len(normalized.split()) == 1:
            q_words = {
                singularize(word)
                for word in q.split()
            }

            if singularize(normalized) in q_words:
                matches.append((len(normalized), column))

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    longest = matches[0][0]

    best = list(dict.fromkeys(
        column
        for length, column in matches
        if length == longest
    ))

    if len(best) == 1:
        return best[0]

    # Prefer exact physical-name occurrence.
    exact = [
        column
        for column in best
        if _phrase_match(q, normalize_text(column))
    ]

    return exact[0] if len(exact) == 1 else None


def _semantic_aliases_for_column(column: str) -> list[str]:
    """Return semantic aliases for a physical column."""

    normalized = normalize_text(column)
    aliases = [normalized]

    for semantic_name, values in COLUMN_ALIASES.items():
        if semantic_name in normalized.split():
            aliases.extend(values)

    return list(dict.fromkeys(
        normalize_text(value)
        for value in aliases
        if value
    ))


def infer_column_by_alias(question, columns):
    """
    Infer a physical column from semantic aliases.

    Only columns that actually exist in the dataset are returned.
    """

    q = normalize_text(question)
    candidates = []

    for column in columns:
        aliases = _semantic_aliases_for_column(str(column))

        matched = [
            alias
            for alias in aliases
            if _phrase_match(q, alias)
        ]

        if matched:
            score = max(len(alias) for alias in matched)
            candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    longest = candidates[0][0]

    best = list(dict.fromkeys(
        column
        for score, column in candidates
        if score == longest
    ))

    return best[0] if len(best) == 1 else None


# =========================================================
# COLUMN INFERENCE
# =========================================================

def infer_numeric_column(question, df, context=None):
    """Infer a numeric measure column."""

    columns = get_numeric_columns(df, context)

    matched = find_column_in_question(
        question,
        columns,
    )

    if matched:
        return matched

    matched = infer_column_by_alias(
        question,
        columns,
    )

    if matched:
        return matched

    if len(columns) == 1:
        return columns[0]

    return None


def infer_group_by(question, df, context=None):
    """Infer a categorical grouping dimension."""

    columns = get_categorical_columns(df, context)

    matched = find_column_in_question(
        question,
        columns,
    )

    if matched:
        return matched

    matched = infer_column_by_alias(
        question,
        columns,
    )

    if matched:
        return matched

    q = normalize_text(question)

    grouping_phrases = [
        "for each",
        "in each",
        "per",
        "by",
        "به تفکیک",
        "بر اساس",
        "براساس",
        "برای هر",
        "در هر",
        "هر کدام",
        "هر یک",
    ]

    if contains_phrase(q, grouping_phrases):
        if len(columns) == 1:
            return columns[0]

    return None


def infer_date_column(df, context=None):
    """Identify an unambiguous date column."""

    columns = get_date_columns(df, context)

    if len(columns) == 1:
        return columns[0]

    preferred_hints = [
        "created date",
        "transaction date",
        "order date",
        "event date",
        "hire date",
        "تاریخ ثبت",
        "تاریخ تراکنش",
        "تاریخ سفارش",
        "تاریخ استخدام",
    ]

    for hint in preferred_hints:
        matches = [
            column
            for column in columns
            if normalize_text(hint) in normalize_text(column)
        ]

        if len(matches) == 1:
            return matches[0]

    return None


# =========================================================
# METRIC RESOLUTION
# =========================================================

def _get_metric_definition(metric_name):
    """Safely retrieve a metric definition."""

    try:
        definition = get_metric(metric_name)

        if isinstance(definition, dict):
            return definition
    except Exception:
        pass

    definition = METRIC_REGISTRY.get(metric_name)

    return definition if isinstance(definition, dict) else None


def _get_available_metric_names(context):
    """Normalize available_metrics into a list of names."""

    available = context.get("available_metrics", [])

    if isinstance(available, dict):
        return list(available.keys())

    if isinstance(available, (list, tuple, set)):
        names = []

        for item in available:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("metric")
                if name:
                    names.append(name)

        return names

    return []


def _metric_aliases(metric_name):
    """Get aliases from registry and local fallback definitions."""

    aliases = [metric_name]

    definition = _get_metric_definition(metric_name)

    if definition:
        value = definition.get("aliases", [])

        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, (list, tuple, set)):
            aliases.extend(value)

        for key in ("label", "name", "description"):
            value = definition.get(key)

            if isinstance(value, str):
                aliases.append(value)

    aliases.extend(METRIC_ALIASES.get(metric_name, []))

    return list(dict.fromkeys(
        normalize_text(value)
        for value in aliases
        if value
    ))


def infer_metric(question, df=None, context=None):
    """
    Infer a registered metric from a question.

    A metric must be available in the current dataset.
    """

    if context is None and df is not None:
        context = get_dataset_context(df)

    if context is None:
        context = {}

    available = set(_get_available_metric_names(context))

    q = normalize_text(question)
    candidates = []

    for metric_name in available:
        for alias in _metric_aliases(metric_name):
            if _phrase_match(q, alias):
                candidates.append((len(alias), metric_name))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    longest = candidates[0][0]

    best = list(dict.fromkeys(
        metric
        for score, metric in candidates
        if score == longest
    ))

    return best[0] if len(best) == 1 else None


def is_analytical_metric(metric_name):
    """Check whether a registry entry is a business metric."""

    if metric_name not in METRIC_REGISTRY:
        return False

    definition = _get_metric_definition(metric_name)

    if not definition:
        return False

    if metric_name in GENERIC_OPERATION_METRICS:
        return False

    return True


def infer_metric_operation(metric_name):
    """Determine the standalone operation for a registered metric."""

    definition = _get_metric_definition(metric_name)

    if not definition:
        return None

    calculation = definition.get("calculation")
    aggregation = definition.get("aggregation")

    if calculation == "boolean_percentage":
        return "return_rate"

    if calculation in {"unique_count", "row_count"}:
        return "count"

    operation_map = {
        "mean": "mean",
        "average": "mean",
        "column_mean": "mean",
        "median": "median",
        "min": "min",
        "minimum": "min",
        "max": "max",
        "maximum": "max",
        "column_sum": "sum",
        "sales_revenue": "sum",
    }

    if aggregation in operation_map:
        return operation_map[aggregation]

    if calculation in operation_map:
        return operation_map[calculation]

    return "sum"


def resolve_available_metric(question, context):
    """Resolve a registered metric available in the dataset."""

    try:
        metric_name = resolve_metric(question)
    except Exception:
        metric_name = None

    available = _get_available_metric_names(context)

    if metric_name in available and is_analytical_metric(metric_name):
        return metric_name

    return infer_metric(
        question,
        context=context,
    )


# =========================================================
# AGGREGATION INFERENCE
# =========================================================

AGGREGATION_PATTERNS = [
    (
        "nunique",
        [
            "distinct", "unique", "different",
            "متفاوت", "مختلف", "غیرتکراری",
            "غیر تکراری", "منحصر به فرد",
            "چند نوع", "تعداد انواع",
        ],
    ),
    (
        "median",
        ["median", "میانه"],
    ),
    (
        "mean",
        ["average", "avg", "mean", "میانگین", "متوسط"],
    ),
    (
        "max",
        [
            "maximum", "max", "highest", "largest",
            "بیشترین", "بالاترین", "حداکثر",
            "بزرگترین", "بزرگ ترین",
        ],
    ),
    (
        "min",
        [
            "minimum", "min", "lowest", "smallest",
            "کمترین", "پایین ترین", "حداقل",
            "کوچکترین", "کوچک ترین",
        ],
    ),
    (
        "count",
        [
            "count", "number of", "تعداد",
            "شمارش", "چند مورد",
        ],
    ),
    (
        "sum",
        [
            "sum", "total", "جمع", "مجموع",
            "جمع کل",
        ],
    ),
]


def infer_aggregation(question):
    """
    Infer an aggregation.

    Distinct-count patterns are evaluated before ordinary count.
    """

    q = normalize_text(question)
    matches = []

    for aggregation, phrases in AGGREGATION_PATTERNS:
        for phrase in phrases:
            if _phrase_match(q, phrase):
                matches.append(
                    (len(normalize_text(phrase)), aggregation)
                )

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    longest = matches[0][0]

    best = list(dict.fromkeys(
        aggregation
        for score, aggregation in matches
        if score == longest
    ))

    return best[0] if len(best) == 1 else None


def infer_simple_operation(question, df, context=None):
    """Infer a generic statistical operation."""

    if has_grouping_request(question):
        return None

    if is_ranking_request(question):
        return None

    aggregation = infer_aggregation(question)

    if aggregation is None:
        return None

    if aggregation == "nunique":
        column = find_column_in_question(
            question,
            list(df.columns),
        )

        if column is None:
            column = infer_column_by_alias(
                question,
                list(df.columns),
            )

        if column is None:
            return None

        query = create_base_query(question)
        query.update({
            "operation": "count",
            "column": column,
            "aggregation": "nunique",
        })

        return query

    column = infer_numeric_column(
        question,
        df,
        context,
    )

    query = create_base_query(question)

    if aggregation == "count" and column is None:
        query.update({
            "operation": "count",
            "column": None,
            "aggregation": "count",
        })
        return query

    if column is None:
        return None

    query.update({
        "operation": aggregation,
        "column": column,
        "aggregation": aggregation,
    })

    return query


# =========================================================
# QUESTION PATTERN DETECTION
# =========================================================

def infer_top_k(question):
    """Extract Top N from Persian or English questions."""

    q = normalize_text(question)

    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\blast\s+(\d+)\b",
        r"(\d+)\s*(?:تای|تا|مورد)",
        r"(\d+)\s*(?:top|برتر)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            value = int(match.group(1))

            if 1 <= value <= MAX_TOP_K:
                return value

    return None


def has_grouping_request(question):
    """Detect explicit and implicit grouping requests."""

    q = normalize_text(question)

    english_patterns = [
        r"\bby\b",
        r"\bfor each\b",
        r"\bin each\b",
        r"\bper\b",
        r"\bgrouped by\b",
        r"\bfor every\b",
    ]

    if any(
        re.search(pattern, q)
        for pattern in english_patterns
    ):
        return True

    phrases = [
        "به تفکیک",
        "بر اساس",
        "براساس",
        "برای هر",
        "در هر",
        "هر محصول",
        "هر شهر",
        "هر دسته",
        "هر مشتری",
        "هر کارمند",
        "هر منطقه",
        "هر دپارتمان",
        "هر بخش",
        "هر واحد",
        "هر گروه",
        "گروه بندی",
        "گروه‌بندی",
    ]

    return contains_phrase(q, phrases)


def is_ranking_request(question):
    """Detect category ranking questions."""

    phrases = [
        "which",
        "highest",
        "lowest",
        "top",
        "best",
        "worst",
        "rank",
        "ranking",
        "most",
        "least",
        "کدام",
        "چه دپارتمانی",
        "کدام بخش",
        "بالاترین",
        "بیشترین",
        "کمترین",
        "پایین ترین",
        "بهترین",
        "بدترین",
        "رتبه بندی",
        "رتبه‌بندی",
    ]

    return contains_phrase(question, phrases)


def infer_sort_direction(question):
    """Infer ascending or descending order."""

    q = normalize_text(question)

    if contains_phrase(q, [
        "lowest", "smallest", "ascending", "least",
        "کمترین", "پایین ترین", "صعودی", "بدترین",
    ]):
        return "asc"

    if contains_phrase(q, [
        "highest", "largest", "top", "descending",
        "most", "بیشترین", "بالاترین",
        "نزولی", "بهترین",
    ]):
        return "desc"

    return None


# =========================================================
# DISTINCT COUNT
# =========================================================

def is_distinct_count_request(question):
    """Detect distinct-count intent."""

    q = normalize_text(question)

    distinct_terms = [
        "distinct",
        "unique",
        "different",
        "متفاوت",
        "مختلف",
        "غیرتکراری",
        "غیر تکراری",
        "منحصر به فرد",
        "چند نوع",
        "تعداد انواع",
    ]

    count_terms = [
        "count",
        "number of",
        "how many",
        "چند",
        "تعداد",
        "شمارش",
    ]

    return (
        contains_phrase(q, distinct_terms)
        and contains_phrase(q, count_terms)
    )


def is_total_row_count_request(question):
    """Detect requests for the total number of dataset rows."""

    phrases = [
        "تعداد کل رکورد",
        "تعداد رکوردها",
        "چند رکورد",
        "تعداد ردیف",
        "تعداد کل ردیف",
        "تعداد کل داده",
        "تعداد کل سطر",
        "total records",
        "number of rows",
        "row count",
        "total rows",
        "how many records",
    ]

    return contains_phrase(question, phrases)

def infer_distinct_column(question, df, context=None):
    """Resolve the column for a distinct-count request."""

    columns = list(df.columns)

    # 1. Prefer an explicitly mentioned physical column.
    column = find_column_in_question(
        question,
        columns,
    )

    if column:
        return column

    # 2. Resolve semantic aliases while ignoring generic
    # count/aggregation words as column hints.
    q = normalize_text(question)

    column_hints = re.sub(
        r"\b(count|number of|how many)\b",
        " ",
        q,
    )

    column_hints = re.sub(
        r"تعداد|شمارش|چند",
        " ",
        column_hints,
    )

    column_hints = " ".join(column_hints.split())

    column = infer_column_by_alias(
        column_hints,
        columns,
    )

    if column:
        return column

    # 3. Use categorical fallback only when unambiguous.
    categorical = get_categorical_columns(
        df,
        context,
    )

    if len(categorical) == 1:
        return categorical[0]

    return None

def build_distinct_count_query(question, df, context=None):
    """Build deterministic distinct-count query."""

    if not is_distinct_count_request(question):
        return None

    column = infer_distinct_column(
        question,
        df,
        context,
    )

    if column is None:
        return None

    query = create_base_query(question)

    query.update({
        "operation": "count",
        "column": column,
        "aggregation": "nunique",
    })

    return query


def build_total_count_query(question):
    """Build deterministic total-row-count query."""

    if not is_total_row_count_request(question):
        return None

    query = create_base_query(question)

    query.update({
        "operation": "count",
        "column": None,
        "aggregation": "count",
    })

    return query


# =========================================================
# QUERY STRUCTURE
# =========================================================

def create_base_query(question):
    """Create the standard query structure."""

    return {
        "operation": None,
        "metric": None,
        "column": None,
        "group_by": None,
        "aggregation": None,
        "filter": None,
        "sort": None,
        "top_k": None,
        "date_column": None,
        "granularity": None,
        "metrics": None,
        "ranking": None,
        "second_column": None,
        "question": question,
    }


def build_metric_query(
    question,
    metric_name,
    df,
    context,
):
    """Build a query using a registered business metric."""

    definition = _get_metric_definition(metric_name)

    if not definition:
        return None

    operation = infer_metric_operation(metric_name)

    if operation is None:
        return None

    query = create_base_query(question)

    query["metric"] = metric_name

    calculation = definition.get("calculation")
    required_columns = definition.get("required_columns") or []

    if calculation == "unique_count" and required_columns:
        query["column"] = required_columns[0]
    else:
        query["column"] = metric_name

    calculation = definition.get("calculation")
    aggregation = definition.get("aggregation")

    if calculation == "unique_count":
        aggregation = "nunique"

    group_by = infer_group_by(
        question,
        df,
        context,
    )

    top_k = infer_top_k(question)
    ranking_request = is_ranking_request(question)

    grouped = (
        has_grouping_request(question)
        or top_k is not None
        or ranking_request
    )

    if grouped:
        if group_by is None:
            return None

        sort = infer_sort_direction(question)

        if ranking_request and sort is None:
            sort = "desc"

        if ranking_request and top_k is None:
            top_k = 1

        query.update({
            "operation": "groupby",
            "group_by": group_by,
            "aggregation": aggregation or "sum",
            "sort": sort,
            "top_k": top_k,
        })

        return query

    query.update({
        "operation": operation,
        "aggregation": aggregation,
    })

    return query


# =========================================================
# GROUPED ANALYSIS
# =========================================================

def build_groupby_query(question, df, context):
    """Build a generic grouped-analysis query."""

    if not has_grouping_request(question) and not is_ranking_request(question):
        return None

    group_by = infer_group_by(
        question,
        df,
        context,
    )

    # Handle implicit grouping in ranking questions:
    # "Which department has the highest average performance?"
    if group_by is None and is_ranking_request(question):
        group_by = infer_column_by_alias(
            question,
            get_categorical_columns(df, context),
        )

    if group_by is None:
        return None

    metric = resolve_available_metric(
        question,
        context,
    )

    if metric is None:
        column = infer_numeric_column(
            question,
            df,
            context,
        )
    else:
        column = metric

    aggregation = infer_aggregation(question)

    if aggregation == "nunique":
        # Distinct-count grouped by another dimension.
        if column is None:
            column = infer_distinct_column(
                question,
                df,
                context,
            )

    if column is None:
        # Count records per group.
        if contains_phrase(question, [
            "تعداد",
            "count",
            "number of",
            "چند",
        ]):
            aggregation = "count"
        else:
            return None

    if aggregation is None:
        if metric:
            definition = _get_metric_definition(metric)
            aggregation = (
                definition.get("aggregation")
                if definition
                else None
            )

        if aggregation is None:
            aggregation = "sum"

    query = create_base_query(question)

    query.update({
        "operation": "groupby",
        "metric": metric,
        "column": column,
        "group_by": group_by,
        "aggregation": aggregation,
    })

    if is_ranking_request(question):
        query["sort"] = infer_sort_direction(question) or "desc"
        query["top_k"] = infer_top_k(question) or 1

    elif infer_top_k(question) is not None:
        query["sort"] = infer_sort_direction(question) or "desc"
        query["top_k"] = infer_top_k(question)

    return query


# =========================================================
# RANKING
# =========================================================

def build_ranking_query(question, df, context):
    """Build a ranking query for categorical dimensions."""

    if not is_ranking_request(question):
        return None

    group_by = infer_group_by(
        question,
        df,
        context,
    )

    if group_by is None:
        group_by = infer_column_by_alias(
            question,
            get_categorical_columns(df, context),
        )

    if group_by is None:
        return None

    metric = resolve_available_metric(
        question,
        context,
    )

    if metric is not None:
        column = metric
        definition = _get_metric_definition(metric)
        aggregation = (
            definition.get("aggregation")
            if definition
            else None
        )

        if definition and definition.get("calculation") == "unique_count":
            aggregation = "nunique"

    else:
        column = infer_numeric_column(
            question,
            df,
            context,
        )
        aggregation = None

    if column is None:
        return None

    requested_aggregation = infer_aggregation(question)

    if requested_aggregation in {
        "mean", "median", "sum", "min", "max",
    }:
        aggregation = requested_aggregation

    if aggregation is None:
        aggregation = "mean"

    if aggregation in {"count", "nunique"} and metric is None:
        # Count ranking is valid when a column was identified.
        pass

    query = create_base_query(question)

    query.update({
        "operation": "groupby",
        "metric": metric,
        "column": column,
        "group_by": group_by,
        "aggregation": aggregation,
        "sort": infer_sort_direction(question) or "desc",
        "top_k": infer_top_k(question) or 1,
    })

    return query


# =========================================================
# FILTER DETECTION
# =========================================================

FILTER_OPERATOR_PATTERNS = [
    (r"\b(greater than|more than|above|over)\b", "gt"),
    (r"\b(at least|greater than or equal to)\b", "gte"),
    (r"\b(less than|below|under)\b", "lt"),
    (r"\b(at most|less than or equal to)\b", "lte"),
    (r"\b(not equal to|is not|does not equal)\b", "ne"),
    (r"\b(equal to|equals|is)\b", "eq"),
]

PERSIAN_FILTER_OPERATOR_PATTERNS = [
    (["بیشتر از", "بزرگتر از", "بالاتر از"], "gt"),
    (["حداقل", "بزرگتر یا مساوی"], "gte"),
    (["کمتر از", "کوچکتر از", "پایین تر از"], "lt"),
    (["حداکثر", "کوچکتر یا مساوی"], "lte"),
    (["مساوی نیست", "برابر نیست"], "ne"),
    (["مساوی", "برابر با"], "eq"),
]


def infer_filter_operator(question):
    """Infer a filter comparison operator."""

    q = normalize_text(question)

    for pattern, operator in FILTER_OPERATOR_PATTERNS:
        if re.search(pattern, q):
            return operator

    for phrases, operator in PERSIAN_FILTER_OPERATOR_PATTERNS:
        if contains_phrase(q, phrases):
            return operator

    if contains_phrase(q, [
        "contains", "includes", "شامل", "حاوی",
    ]):
        return "contains"

    return None


def has_filter_request(question):
    """Detect filtering intent."""

    phrases = [
        "where",
        "whose",
        "with",
        "having",
        "filter",
        "only",
        "greater than",
        "less than",
        "equal to",
        "not equal to",
        "بیشتر از",
        "کمتر از",
        "بزرگتر از",
        "کوچکتر از",
        "حداقل",
        "حداکثر",
        "فقط",
        "دارای",
        "شامل",
        "برابر با",
        "مساوی",
    ]

    return contains_phrase(question, phrases)


def infer_filter_value(question, column, df):
    """Infer a filter value using actual dataset values first."""

    q = normalize_text(question)

    values = df[column].dropna().unique().tolist()
    matches = []

    for value in values:
        normalized = normalize_text(value)

        if normalized and _phrase_match(q, normalized):
            matches.append(value)

    if matches:
        return max(
            matches,
            key=lambda value: len(normalize_text(value)),
        )

    numeric_match = re.search(
        r"-?\d+(?:\.\d+)?",
        q,
    )

    if numeric_match:
        try:
            number = float(numeric_match.group(0))

            if number.is_integer():
                return int(number)

            return number

        except ValueError:
            pass

    return None


def build_simple_filter_query(question, df, context):
    """Build a filter query when the condition is clear."""

    if not has_filter_request(question):
        return None

    operator = infer_filter_operator(question)

    if operator is None:
        return None

    column = find_column_in_question(
        question,
        list(df.columns),
    )

    if column is None:
        column = infer_column_by_alias(
            question,
            list(df.columns),
        )

    if column is None:
        return None

    value = infer_filter_value(
        question,
        column,
        df,
    )

    if value is None:
        return None

    query = create_base_query(question)

    query.update({
        "operation": "filter",
        "filter": {
            "column": column,
            "operator": operator,
            "value": value,
        },
        "top_k": infer_top_k(question),
    })

    return query


# =========================================================
# TREND ANALYSIS
# =========================================================

TREND_WORDS = [
    "trend",
    "over time",
    "timeline",
    "time series",
    "historical trend",
    "روند",
    "تغییرات",
    "در طول زمان",
    "طی زمان",
    "سیر تغییرات",
]


def is_trend_request(question):
    """Detect explicit time-based analysis."""

    return contains_phrase(
        question,
        TREND_WORDS,
    )


def infer_granularity(question):
    """Infer time granularity. Monthly is the default."""

    patterns = [
        (["daily", "per day", "روزانه", "هر روز"], "day"),
        (["weekly", "per week", "هفتگی", "هر هفته"], "week"),
        (["monthly", "per month", "ماهانه", "هر ماه"], "month"),
        (["quarterly", "quarter", "فصلی", "هر فصل"], "quarter"),
        (["yearly", "per year", "سالانه", "هر سال"], "year"),
    ]

    for phrases, granularity in patterns:
        if contains_phrase(question, phrases):
            return granularity

    return "month"


def infer_trend_metrics(question, context):
    """
    Select relevant metrics for trend analysis.

    Uses only metrics available in the current dataset.
    """

    available = _get_available_metric_names(context)

    definitions = context.get("metric_definitions", {})
    domain = context.get("domain", {}).get("name")

    metric_names = [
        name
        for name in available
        if is_analytical_metric(name)
    ]

    selected = []

    for metric_name in metric_names:
        definition = (
            definitions.get(metric_name)
            or _get_metric_definition(metric_name)
            or {}
        )

        aliases = _metric_aliases(metric_name)
        aliases.extend(definition.get("aliases", []))

        if contains_phrase(question, aliases):
            selected.append(metric_name)

    if selected:
        return list(dict.fromkeys(selected))

    domain_preferences = {
        "sales": {
            "revenue": [
                "sales", "sales trend", "فروش",
                "روند فروش", "درآمد فروش",
            ],
            "quantity": [
                "units sold", "items sold",
                "مقدار فروش", "تعداد اقلام فروخته شده",
            ],
        },
        "finance": {
            "revenue": [
                "income", "revenue", "درآمد",
            ],
        },
        "hr": {
            "employee_count": [
                "employee count", "number of employees",
                "headcount", "تعداد کارکنان",
                "تعداد کارمندان",
            ],
            "average_salary": [
                "salary", "average salary",
                "حقوق", "میانگین حقوق",
            ],
        },
    }

    for metric_name, aliases in domain_preferences.get(domain, {}).items():
        if metric_name in metric_names and contains_phrase(question, aliases):
            return [metric_name]

    if contains_phrase(question, [
        "all metrics",
        "all measures",
        "تمام معیارها",
        "همه معیارها",
        "تمام شاخص ها",
        "همه شاخص ها",
    ]):
        result = []

        for name in metric_names:
            definition = (
                definitions.get(name)
                or _get_metric_definition(name)
                or {}
            )

            if definition.get("calculation") in {
                "column_sum",
                "sales_revenue",
                "unique_count",
                "row_count",
                "column_mean",
            }:
                result.append(name)

        return result

    additive_metrics = []

    for metric_name in metric_names:
        definition = (
            definitions.get(metric_name)
            or _get_metric_definition(metric_name)
            or {}
        )

        if definition.get("calculation") in {
            "column_sum",
            "sales_revenue",
        }:
            additive_metrics.append(metric_name)

    if len(additive_metrics) == 1:
        return additive_metrics

    return []


def infer_trend_analysis(question, df, context=None):
    """Build a general-purpose time-trend query."""

    if not is_trend_request(question):
        return None

    date_column = infer_date_column(
        df,
        context,
    )

    if date_column is None:
        return None

    metrics = infer_trend_metrics(
        question,
        context or {},
    )

    if not metrics:
        return None

    query = create_base_query(question)

    query.update({
        "operation": "trend_by_date",
        "date_column": date_column,
        "granularity": infer_granularity(question),
        "metrics": metrics,
        "ranking": infer_sort_direction(question),
        "sort": "asc",
    })

    return query


# =========================================================
# OLLAMA PROMPT
# =========================================================

def build_parser_prompt(question, df, context):
    """Build a context-aware prompt for Ollama."""

    columns = list(df.columns)
    sample_values = {}

    for column in columns:
        sample_values[column] = (
            df[column]
            .dropna()
            .astype(str)
            .unique()[:8]
            .tolist()
        )

    prompt = f"""
You are a professional general-purpose data analysis query interpreter.

Convert the user's question into exactly one safe JSON query.

DATASET DOMAIN:
{context.get("domain", {}).get("name")}

DATASET COLUMNS:
{json.dumps(columns, ensure_ascii=False)}

COLUMN SCHEMA:
{json.dumps(context.get("schema", []), ensure_ascii=False, default=str)}

SAMPLE VALUES:
{json.dumps(sample_values, ensure_ascii=False, default=str)}

AVAILABLE METRICS:
{json.dumps(_get_available_metric_names(context), ensure_ascii=False)}

METRIC DEFINITIONS:
{json.dumps(context.get("metric_definitions", {}), ensure_ascii=False, default=str)}

ALLOWED OPERATIONS:
{json.dumps(sorted(ALLOWED_OPERATIONS))}

ALLOWED AGGREGATIONS:
{json.dumps(sorted(ALLOWED_AGGREGATIONS))}

ALLOWED FILTER OPERATORS:
{json.dumps(sorted(ALLOWED_FILTER_OPERATORS))}

Return this JSON structure:

{{
    "operation": null,
    "metric": null,
    "column": null,
    "group_by": null,
    "aggregation": null,
    "filter": null,
    "sort": null,
    "top_k": null,
    "date_column": null,
    "granularity": null,
    "metrics": null,
    "ranking": null,
    "second_column": null,
    "question": ""
}}

RULES:
1. Use only actual dataset columns and available metrics.
2. Never invent columns, metrics or filter values.
3. Use exact physical column names.
4. Use exact registered metric identifiers.
5. Use operation "count" and aggregation "nunique"
   for distinct-count questions.
6. Use operation "count" and column null
   for total row-count questions.
7. Use operation "groupby" for grouped analysis.
8. Use categorical columns for group_by.
9. Use "trend_by_date" for time-trend questions.
10. Trend queries require date_column and metrics.
11. Use "month" when no trend interval is specified.
12. Use "filter" for filtering requests.
13. Filter operators must be from the allowed list.
14. For highest or best ranking use sort "desc".
15. For lowest or worst ranking use sort "asc".
16. For one best or worst result use top_k 1.
17. Return JSON only.

USER QUESTION:
{question}
"""

    return prompt


# =========================================================
# OLLAMA RESPONSE PARSING
# =========================================================

def extract_json_response(response):
    """Extract JSON from Ollama response."""

    content = None

    if hasattr(response, "message"):
        message = response.message

        if hasattr(message, "content"):
            content = message.content

    if content is None and isinstance(response, dict):
        content = (
            response.get("message", {})
            .get("content")
        )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response.")

    content = content.strip()

    try:
        result = json.loads(content)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL,
    )

    if match:
        try:
            result = json.loads(match.group(0))

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError("Ollama returned invalid JSON.")


def get_ollama_client():
    """Create Ollama client lazily."""

    if Client is None:
        return None

    try:
        return Client(host=OLLAMA_HOST)
    except Exception:
        return None


def interpret_with_ollama(question, df, context):
    """Use Ollama only when deterministic parsing is insufficient."""

    ollama_client = get_ollama_client()

    if ollama_client is None:
        return None

    prompt = build_parser_prompt(
        question,
        df,
        context,
    )

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict data-analysis query interpreter. "
                        "Return one valid JSON object only. "
                        "Never invent columns, metrics or filter values."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format="json",
            options={
                "temperature": TEMPERATURE,
            },
        )

        result = extract_json_response(response)

        result.setdefault("question", question)

        return result

    except Exception:
        return None


# =========================================================
# QUERY REPAIR
# =========================================================

def repair_query(result, question, df, context):
    """
    Repair only fields that can be inferred safely.

    Does not replace unknown columns or metrics with guesses.
    """

    if not isinstance(result, dict):
        return None

    result = dict(result)
    operation = result.get("operation")

    if operation == "groupby":
        if result.get("group_by") is None:
            result["group_by"] = infer_group_by(
                question,
                df,
                context,
            )

        if result.get("column") is None:
            metric = resolve_available_metric(
                question,
                context,
            )

            if metric:
                result["metric"] = metric
                result["column"] = metric
            else:
                result["column"] = infer_numeric_column(
                    question,
                    df,
                    context,
                )

        if result.get("aggregation") is None:
            result["aggregation"] = infer_aggregation(question)

        if is_ranking_request(question):
            result["sort"] = (
                result.get("sort")
                or infer_sort_direction(question)
                or "desc"
            )

            result["top_k"] = (
                result.get("top_k")
                or infer_top_k(question)
                or 1
            )

    elif operation == "trend_by_date":
        if result.get("date_column") is None:
            result["date_column"] = infer_date_column(
                df,
                context,
            )

        if result.get("granularity") is None:
            result["granularity"] = infer_granularity(question)

        if not result.get("metrics"):
            result["metrics"] = infer_trend_metrics(
                question,
                context,
            )

    elif operation == "filter":
        condition = result.get("filter")

        if isinstance(condition, dict):
            condition = dict(condition)

            if condition.get("column") is None:
                condition["column"] = find_column_in_question(
                    question,
                    list(df.columns),
                )

            if condition.get("operator") is None:
                condition["operator"] = infer_filter_operator(
                    question
                )

            column = condition.get("column")

            if condition.get("value") is None and column in df.columns:
                condition["value"] = infer_filter_value(
                    question,
                    column,
                    df,
                )

            result["filter"] = condition

    return result


# =========================================================
# QUERY VALIDATION
# =========================================================

def validate_column_reference(
    value,
    columns,
    available_metrics,
    field_name,
):
    """Validate a physical column or registered metric."""

    if value is None:
        return

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string."
        )

    if value not in columns and value not in available_metrics:
        raise ValueError(
            f"Invalid {field_name}: {value}"
        )


def validate_result(result, columns, context=None):
    """
    Validate and normalize a query.

    Preserves the query schema expected by the existing
    query_engine and assistant modules.
    """

    if not isinstance(result, dict):
        raise ValueError("Query must be a dictionary.")

    context = context or {}

    available_metrics = _get_available_metric_names(context)

    result = dict(result)

    defaults = {
        "metric": None,
        "column": None,
        "group_by": None,
        "aggregation": None,
        "filter": None,
        "sort": None,
        "top_k": None,
        "date_column": None,
        "granularity": None,
        "metrics": None,
        "ranking": None,
        "second_column": None,
        "question": "",
    }

    for field, default in defaults.items():
        result.setdefault(field, default)

    # -----------------------------------------------------
    # Operation
    # -----------------------------------------------------

    operation_aliases = {
        "average": "mean",
        "avg": "mean",
        "minimum": "min",
        "maximum": "max",
    }

    operation = result.get("operation")

    if not isinstance(operation, str):
        raise ValueError("A valid operation is required.")

    operation = operation.strip().lower()
    operation = operation_aliases.get(operation, operation)

    result["operation"] = operation

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # -----------------------------------------------------
    # Metric
    # -----------------------------------------------------

    metric = result.get("metric")

    if metric is not None:
        if not isinstance(metric, str):
            raise ValueError("Metric must be a string.")

        if metric not in available_metrics:
            raise ValueError(
                f"Unavailable metric: {metric}"
            )

    # -----------------------------------------------------
    # Column references
    # -----------------------------------------------------

    validate_column_reference(
        result.get("column"),
        columns,
        available_metrics,
        "column",
    )

    for field in (
        "group_by",
        "date_column",
        "second_column",
    ):
        value = result.get(field)

        if value is not None:
            if not isinstance(value, str) or value not in columns:
                raise ValueError(
                    f"Invalid column in {field}: {value}"
                )

    # -----------------------------------------------------
    # Aggregation
    # -----------------------------------------------------

    aggregation_aliases = {
        "average": "mean",
        "avg": "mean",
        "minimum": "min",
        "maximum": "max",
        "distinct": "nunique",
        "unique": "nunique",
    }

    aggregation = result.get("aggregation")

    if aggregation is not None:
        if not isinstance(aggregation, str):
            raise ValueError(
                "Aggregation must be a string."
            )

        aggregation = aggregation.strip().lower()
        aggregation = aggregation_aliases.get(
            aggregation,
            aggregation,
        )

        if aggregation not in ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation: {aggregation}"
            )

        result["aggregation"] = aggregation

    # -----------------------------------------------------
    # Count
    # -----------------------------------------------------

    if operation == "count":
        column = result.get("column")
        aggregation = result.get("aggregation")

        if aggregation == "nunique":
            if column is None or column not in columns:
                raise ValueError(
                    "Distinct count requires a physical dataset column."
                )

        elif column is None:
            result["aggregation"] = "count"

        elif aggregation is None:
            result["aggregation"] = "count"

    # -----------------------------------------------------
    # Simple statistical operations
    # -----------------------------------------------------

    if operation in {
        "sum", "mean", "median", "min", "max",
    }:
        if result.get("column") is None:
            raise ValueError(
                f"Operation '{operation}' requires a column or metric."
            )

    # -----------------------------------------------------
    # Sort and ranking
    # -----------------------------------------------------

    for field in ("sort", "ranking"):
        value = result.get(field)

        if value is not None and value not in ("asc", "desc"):
            raise ValueError(
                f"{field} must be asc, desc or null."
            )

    # -----------------------------------------------------
    # Top K
    # -----------------------------------------------------

    top_k = result.get("top_k")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")

        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be between 1 and {MAX_TOP_K}."
            )

        if operation not in {"groupby", "filter"}:
            raise ValueError(
                "top_k is only supported with groupby or filter."
            )

    # -----------------------------------------------------
    # Grouped analysis
    # -----------------------------------------------------

    if operation == "groupby":
        if result.get("group_by") is None:
            raise ValueError(
                "Grouping column is required."
            )

        if result.get("column") is None:
            if result.get("aggregation") != "count":
                raise ValueError(
                    "Metric or column is required for grouping."
                )

        if result.get("aggregation") is None:
            definition = (
                _get_metric_definition(metric)
                if metric
                else None
            )

            result["aggregation"] = (
                definition.get("aggregation")
                if definition
                else "sum"
            )

        definition = (
            _get_metric_definition(metric)
            if metric
            else None
        )

        if definition and definition.get("calculation") == "unique_count":
            result["aggregation"] = "nunique"

        if result.get("aggregation") == "nunique":
            if result.get("column") is None:
                raise ValueError(
                    "Grouped distinct count requires a column."
                )

    # -----------------------------------------------------
    # Filter
    # -----------------------------------------------------

    if operation == "filter":
        condition = result.get("filter")

        if not isinstance(condition, dict):
            raise ValueError(
                "A filter condition is required."
            )

        condition = dict(condition)

        filter_column = condition.get("column")
        operator = condition.get("operator")

        if filter_column not in columns:
            raise ValueError("Invalid filter column.")

        if operator not in ALLOWED_FILTER_OPERATORS:
            raise ValueError("Invalid filter operator.")

        if "value" not in condition or condition["value"] is None:
            raise ValueError("A non-null filter value is required.")

        if operator == "contains":
            condition["value"] = str(condition["value"])

        result["filter"] = condition

    # -----------------------------------------------------
    # Return rate
    # -----------------------------------------------------

    if operation == "return_rate":
        if metric is not None:
            definition = _get_metric_definition(metric)

            if not definition:
                raise ValueError(
                    "Invalid return-rate metric."
                )

            if definition.get("calculation") != "boolean_percentage":
                raise ValueError(
                    "The selected metric is not a Boolean percentage metric."
                )

            required = definition.get("required_columns", [])

            for column in required:
                if column not in columns:
                    raise ValueError(
                        f"Required Boolean column is missing: {column}"
                    )

        elif result.get("column") not in columns:
            raise ValueError(
                "A valid Boolean status column is required."
            )

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    if operation == "trend_by_date":
        date_column = result.get("date_column")

        if date_column not in columns:
            raise ValueError(
                "A valid date column is required."
            )

        granularity = result.get("granularity") or "month"

        if granularity not in ALLOWED_GRANULARITIES:
            raise ValueError(
                "Invalid trend granularity."
            )

        result["granularity"] = granularity

        metrics = result.get("metrics")

        if not isinstance(metrics, list) or not metrics:
            raise ValueError(
                "At least one trend metric is required."
            )

        for trend_metric in metrics:
            if trend_metric not in available_metrics:
                raise ValueError(
                    f"Unavailable trend metric: {trend_metric}"
                )

            definition = _get_metric_definition(trend_metric)

            if not definition:
                raise ValueError(
                    f"Invalid trend metric: {trend_metric}"
                )

            if definition.get("calculation") == "boolean_percentage":
                raise ValueError(
                    "Boolean percentage metrics are not supported in trend analysis."
                )

    return result


# =========================================================
# DETERMINISTIC PARSING
# =========================================================

def build_deterministic_query(question, df, context):
    """
    Try deterministic parsing before Ollama.

    Priority:
    1. Distinct count.
    2. Total row count.
    3. Ranking.
    4. Trend.
    5. Grouped analysis.
    6. Filter.
    7. Registered metric.
    8. Generic statistical operation.
    """

    # 1. Distinct count.
    query = build_distinct_count_query(
        question,
        df,
        context,
    )

    if query is not None:
        return query

    # 2. Total row count.
    query = build_total_count_query(question)

    if query is not None:
        return query

    # 3. Ranking.
    query = build_ranking_query(
        question,
        df,
        context,
    )

    if query is not None:
        return query

    # 4. Trend.
    query = infer_trend_analysis(
        question,
        df,
        context,
    )

    if query is not None:
        return query

    # 5. Grouped analysis.
    query = build_groupby_query(
        question,
        df,
        context,
    )

    if query is not None:
        return query

    # 6. Filtering.
    query = build_simple_filter_query(
        question,
        df,
        context,
    )

    if query is not None:
        return query

    # 7. Registered metric.
    metric = resolve_available_metric(
        question,
        context,
    )

    if metric is not None:
        query = build_metric_query(
            question,
            metric,
            df,
            context,
        )

        if query is not None:
            return query

    # 8. Generic statistical operation.
    return infer_simple_operation(
        question,
        df,
        context,
    )



def infer_registered_metric(question):
    """
    Infer a metric from the full registry,
    regardless of its availability in the current dataset.
    """

    candidates = []

    for metric_name in METRIC_REGISTRY:
        for alias in _metric_aliases(metric_name):
            if _phrase_match(question, alias):
                candidates.append((len(alias), metric_name))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    longest = candidates[0][0]

    best = list(dict.fromkeys(
        metric
        for score, metric in candidates
        if score == longest
    ))

    return best[0] if len(best) == 1 else None


def validate_ollama_semantics(result, question, df, context):
    """
    Reject LLM queries that conflict with explicit
    metric or column intent in the user's question.
    """

    available_metrics = set(
        _get_available_metric_names(context)
    )

    # Check registered metric intent.
    expected_metric = infer_registered_metric(question)

    if expected_metric is not None:
        if expected_metric not in available_metrics:
            return False

        if result.get("metric") != expected_metric:
            return False

        return True

    # Check explicitly mentioned physical or semantic columns.
    columns = list(df.columns)

    expected_column = find_column_in_question(
        question,
        columns,
    )

    if expected_column is None:
        expected_column = infer_column_by_alias(
            question,
            columns,
        )

    if expected_column is not None:
        if result.get("operation") in {
            "sum", "mean", "median", "min", "max",
        }:
            if result.get("metric") is None:
                return result.get("column") == expected_column

    return True


# =========================================================
# QUERY REPAIR AND VALIDATION PIPELINE
# =========================================================

def parse_with_ollama(question, df, context):
    """Interpret and validate an Ollama-generated query."""

    result = interpret_with_ollama(
        question,
        df,
        context,
    )

    if result is None:
        return None

    if not validate_ollama_semantics(
        result,
        question,
        df,
        context,
    ):
        return None

    result = repair_query(
        result,
        question,
        df,
        context,
    )

    if result is None:
        return None

    try:
        return validate_result(
            result,
            list(df.columns),
            context,
        )
    except (ValueError, TypeError, KeyError):
        return None


# =========================================================
# MAIN PARSER
# =========================================================

def parse_question(question: str, df: pd.DataFrame) -> dict:
    """
    Convert a Persian or English question into a validated query.

    All numerical calculations remain in query_engine.py.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")

    context = get_dataset_context(df)
    columns = list(df.columns)

    # Deterministic parsing has priority.
    deterministic_result = build_deterministic_query(
        question,
        df,
        context,
    )

    if deterministic_result is not None:
        return validate_result(
            deterministic_result,
            columns,
            context,
        )

    # Ollama is only a fallback.
    llm_result = parse_with_ollama(
        question,
        df,
        context,
    )

    if llm_result is not None:
        return llm_result

    raise ValueError(
        "Could not determine a valid analytical operation. "
        "Please specify the metric, column, or operation more clearly."
    )


# =========================================================
# COMPATIBILITY HELPERS
# =========================================================

def parse_intent(question, df):
    """Backward-compatible alias."""

    return parse_question(question, df)


def validate_query(query, df):
    """Backward-compatible validation wrapper."""

    context = get_dataset_context(df)

    return validate_result(
        query,
        list(df.columns),
        context,
    )


def get_query_intent(question, df):
    """Backward-compatible alias."""

    return parse_question(question, df)
