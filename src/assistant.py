
"""
General-Purpose Data Analysis Assistant
=======================================

Responsibilities:
- Coordinate the complete analysis pipeline.
- Validate user questions and datasets.
- Classify exploratory and analytical requests.
- Route exploratory requests to the EDA engine.
- Parse analytical questions into executable queries.
- Execute deterministic numerical calculations.
- Format analytical evidence in Persian or English.
- Use Ollama for classification and evidence interpretation.
- Generate charts for supported analytical requests.
- Preserve original dataset category values.
- Return structured analysis results and execution metadata.

Architecture:

    User Input
        |
    Input Validation
        |
    Intent Classifier
        |
        +--------------------------+
        |                          |
    Exploratory Analysis      Analytical Query
        |                          |
    EDA Engine                Intent Parser
        |                          |
    EDA Evidence              Query Engine
        |                          |
        +------------+-------------+
                     |
                  Evidence
                     |
             Deterministic Formatter
                     |
             LLM Interpretation
                (when available)
                     |
              Chart Generator
                     |
              Structured Result

Numerical calculations are performed deterministically
by the EDA and query engines using Pandas.
"""

import os
import json
import math
import re
import time

import pandas as pd
from src.config import OLLAMA_MODEL
from src.ollama_client import ask_llm

from src.intent_parser import parse_question
from src.query_engine import execute_query
from src.metric_registry import get_metric
from src.chart_generator import generate_trend_chart
from src.intent_classifier import classify_intent
from src.eda_engine import analyze_dataset, build_llm_evidence



# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_OUTPUT_LANGUAGE = os.getenv(
    "OUTPUT_LANGUAGE",
    "auto"
).strip().lower()

MAX_FILTER_DISPLAY = 10

SUPPORTED_LANGUAGES = ("fa", "en")

# Distinguishes an omitted model argument from model=None.
_DEFAULT_MODEL = object()


# =========================================================
# LANGUAGE
# =========================================================

def detect_language(text):
    """Detect Persian or English from text."""

    if re.search(r"[\u0600-\u06FF]", str(text)):
        return "fa"

    return "en"


def resolve_output_language(question):
    """Resolve the language used for the final answer."""

    language = DEFAULT_OUTPUT_LANGUAGE

    if language in ("fa", "persian", "farsi"):
        return "fa"

    if language in ("en", "english"):
        return "en"

    return detect_language(question)


# =========================================================
# TEXT RESOURCES
# =========================================================

MESSAGES = {
    "fa": {
        "analysis_result": "نتیجه تحلیل",
        "group_result": "نتایج به تفکیک",
        "metric": "معیار",
        "group": "گروه‌بندی بر اساس",
        "aggregation": "نوع محاسبه",
        "total_groups": "تعداد گروه‌ها",
        "total": "مقدار کل",
        "count": "تعداد",
        "valid_values": "تعداد مقادیر معتبر",
        "return_rate": "نرخ مرجوعی",
        "total_records": "تعداد کل رکوردهای معتبر",
        "positive_records": "تعداد رکوردهای مثبت",
        "percentage": "درصد",
        "period": "دوره",
        "trend": "تحلیل روند",
        "highest": "بیشترین مقدار",
        "lowest": "کمترین مقدار",
        "changes": "تغییرات",
        "increase": "افزایش",
        "decrease": "کاهش",
        "unchanged": "بدون تغییر",
        "from": "از",
        "to": "به",
        "time_granularity": "دوره زمانی",
        "records": "رکوردها",
        "total_matches": "تعداد رکوردهای منطبق",
        "displayed_records": "تعداد رکوردهای نمایش‌داده‌شده",
        "no_results": "نتیجه‌ای برای نمایش وجود ندارد.",
        "chart_not_supported": "برای این نوع تحلیل نمودار مستقیمی تعریف نشده است.",
        "analysis_error": "خطا در تحلیل داده",
        "total_rows": "تعداد کل ردیف‌ها",
        "sum": "مجموع",
        "mean": "میانگین",
        "median": "میانه",
        "min": "حداقل",
        "max": "حداکثر",
        "count_operation": "تعداد",
        "nunique": "تعداد مقادیر یکتا",
        "unknown": "نامشخص",
    },

    "en": {
        "analysis_result": "Analysis Result",
        "group_result": "Grouped Results",
        "metric": "Metric",
        "group": "Grouped by",
        "aggregation": "Aggregation",
        "total_groups": "Total Groups",
        "total": "Total",
        "count": "Count",
        "valid_values": "Valid Values",
        "return_rate": "Return Rate",
        "total_records": "Total Valid Records",
        "positive_records": "Positive Records",
        "percentage": "Percentage",
        "period": "Period",
        "trend": "Trend Analysis",
        "highest": "Highest Value",
        "lowest": "Lowest Value",
        "changes": "Changes",
        "increase": "Increase",
        "decrease": "Decrease",
        "unchanged": "Unchanged",
        "from": "From",
        "to": "To",
        "time_granularity": "Time Granularity",
        "records": "Records",
        "total_matches": "Total Matching Records",
        "displayed_records": "Displayed Records",
        "no_results": "No results are available.",
        "chart_not_supported": "A direct chart is not defined for this analysis type.",
        "analysis_error": "Data Analysis Error",
        "total_rows": "Total Rows",
        "sum": "Sum",
        "mean": "Average",
        "median": "Median",
        "min": "Minimum",
        "max": "Maximum",
        "count_operation": "Count",
        "nunique": "Unique Count",
        "unknown": "Unknown",
    },
}


def get_text(language, key):
    """Return a localized text resource."""

    language = language if language in MESSAGES else "en"

    return MESSAGES[language].get(
        key,
        key
    )


# =========================================================
# VALUE UTILITIES
# =========================================================

def is_missing(value):
    """Safely detect missing scalar values."""

    try:
        result = pd.isna(value)

        if hasattr(result, "ndim") and result.ndim > 0:
            return False

        return bool(result)

    except (TypeError, ValueError):
        return False


def safe_text(value):
    """Convert values to display-safe text."""

    if value is None or is_missing(value):
        return "-"

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass

    return str(value)


def format_number(value, decimals=2):
    """Format a numeric value without inventing precision."""

    if value is None or is_missing(value):
        return "-"

    try:
        number = float(value)

        if not math.isfinite(number):
            return "-"

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")

    except (TypeError, ValueError, OverflowError):
        return safe_text(value)


def format_percentage(value):
    """Format a percentage value."""

    formatted = format_number(value)

    if formatted == "-":
        return "-"

    return f"{formatted}%"


def format_metric_value(value, metric_name=None):
    """
    Format metric values according to metric metadata.

    Currency is not assumed unless explicitly declared.
    """

    formatted = format_number(value)

    if formatted == "-":
        return formatted

    definition = get_metric(metric_name) if metric_name else None

    if not definition:
        return formatted

    unit = definition.get("unit")

    if unit == "percentage":
        return f"{formatted}%"

    if unit == "currency_usd":
        return f"${formatted}"

    if unit == "currency":
        return formatted

    return formatted


# =========================================================
# COLUMN LABELS
# =========================================================

COLUMN_LABELS = {
    "fa": {
        "city": "شهر",
        "product": "محصول",
        "category": "دسته‌بندی",
        "quantity": "تعداد",
        "unit_price": "قیمت واحد",
        "unit_price_usd": "قیمت واحد (دلار)",
        "price": "قیمت",
        "revenue": "درآمد فروش",
        "sales": "فروش",
        "order_id": "شناسه سفارش",
        "order_date": "تاریخ سفارش",
        "date": "تاریخ",
        "customer": "مشتری",
        "customer_id": "شناسه مشتری",
        "customer_rating": "امتیاز مشتری",
        "payment_method": "روش پرداخت",
        "returned": "مرجوع شده",
        "discount": "تخفیف",
        "discount_pct": "درصد تخفیف",
        "employee": "کارمند",
        "employee_id": "شناسه کارمند",
        "department": "واحد سازمانی",
        "salary": "حقوق",
        "age": "سن",
        "gender": "جنسیت",
        "country": "کشور",
        "region": "منطقه",
        "status": "وضعیت",
        "name": "نام",
        "id": "شناسه",
        "amount": "مبلغ",
        "total": "مجموع",
        "cost": "هزینه",
        "profit": "سود",
        "project": "پروژه",
        "project_id": "شناسه پروژه",
        "task": "وظیفه",
        "duration": "مدت زمان",
        "hours": "ساعت",
        "score": "امتیاز",
        "rating": "رتبه",
    },
    "en": {},
}


def get_column_label(column, language="en"):
    """Return a localized column label."""

    if column is None or column == "":
        return "Column" if language == "en" else "ستون"

    column = str(column)

    translated = COLUMN_LABELS.get(
        language,
        {}
    ).get(column.lower())

    if translated:
        return translated

    return column.replace("_", " ")


# =========================================================
# METRIC LABELS
# =========================================================

def get_metric_label(metric_name, language="en"):
    """Return a metric label."""

    if not metric_name:
        return get_text(language, "metric")

    definition = get_metric(metric_name)

    if language == "fa" and definition:
        return definition.get("label", metric_name)

    return str(metric_name).replace("_", " ").title()


def get_aggregation_label(aggregation, language="en"):
    """Return a localized aggregation label."""

    if not aggregation:
        return ""

    return get_text(
        language,
        str(aggregation).lower()
    )


# =========================================================
# GROUP BY
# =========================================================

def format_groupby_answer(evidence, language="en"):
    results = evidence.get("results", [])

    if not results:
        return get_text(language, "no_results")

    metric = evidence.get("metric")
    group_by = evidence.get("group_by")
    aggregation = evidence.get("aggregation")

    metric_label = get_metric_label(
        metric,
        language
    )

    group_label = get_column_label(
        group_by,
        language
    )

    lines = [
        f"## {get_text(language, 'analysis_result')}",
        "",
        f"**{get_text(language, 'metric')}:** {metric_label}",
        f"**{get_text(language, 'group')}:** {group_label}",
    ]

    if aggregation:
        lines.append(
            f"**{get_text(language, 'aggregation')}:** "
            f"{get_aggregation_label(aggregation, language)}"
        )

    lines.extend([
        "",
        f"### {get_text(language, 'group_result')}",
        "",
    ])

    for record in results:
        group_value = safe_text(
            record.get(group_by)
        )

        value = format_metric_value(
            record.get("result"),
            metric
        )

        lines.append(
            f"- **{group_value}:** {value}"
        )

    total_groups = evidence.get("total_groups")

    if total_groups is not None:
        lines.extend([
            "",
            f"**{get_text(language, 'total_groups')}:** "
            f"{format_number(total_groups)}",
        ])

    return "\n".join(lines)


# =========================================================
# STATISTICS
# =========================================================

def format_statistic_answer(evidence, language="en"):
    operation = evidence.get("operation")
    metric = evidence.get("metric")
    column = evidence.get("column")

    label = (
        get_metric_label(metric, language)
        if metric
        else get_column_label(column, language)
    )

    operation_label = get_aggregation_label(
        operation,
        language
    )

    value = format_metric_value(
        evidence.get("result"),
        metric
    )

    lines = [
        f"## {get_text(language, 'analysis_result')}",
        "",
        f"**{get_text(language, 'metric')}:** {label}",
        f"**{operation_label}:** {value}",
    ]

    valid_values = evidence.get("valid_values")

    if valid_values is not None:
        lines.append(
            f"**{get_text(language, 'valid_values')}:** "
            f"{format_number(valid_values)}"
        )

    return "\n".join(lines)


# =========================================================
# COUNT
# =========================================================

def format_count_answer(evidence, language="en"):
    count = evidence.get("count", 0)

    metric = evidence.get("metric")
    column = evidence.get("column")

    if metric:
        label = get_metric_label(
            metric,
            language
        )
    elif column:
        label = get_column_label(
            column,
            language
        )
    else:
        label = get_text(
            language,
            "total_rows"
        )

    return "\n".join([
        f"## {get_text(language, 'analysis_result')}",
        "",
        f"**{label}:** {format_number(count)}",
    ])


# =========================================================
# RETURN RATE
# =========================================================

def format_return_rate_answer(evidence, language="en"):
    results = evidence.get("results")

    if results is not None:
        if not results:
            return get_text(language, "no_results")

        group_by = evidence.get("group_by")

        lines = [
            f"## {get_text(language, 'return_rate')}",
            "",
        ]

        if group_by:
            lines.append(
                f"**{get_text(language, 'group')}:** "
                f"{get_column_label(group_by, language)}"
            )

        lines.append("")

        for record in results:
            group_value = (
                safe_text(record.get(group_by))
                if group_by
                else None
            )

            if group_value is not None:
                lines.append(f"### {group_value}")

            total = record.get("total_records", 0)
            positive = record.get("positive_records", 0)

            percentage = record.get(
                "return_rate_pct",
                record.get("percentage")
            )

            lines.append(
                f"- {get_text(language, 'total_records')}: "
                f"{format_number(total)}"
            )

            lines.append(
                f"- {get_text(language, 'positive_records')}: "
                f"{format_number(positive)}"
            )

            lines.append(
                f"- **{get_text(language, 'return_rate')}: "
                f"{format_percentage(percentage)}**"
            )

            lines.append("")

        return "\n".join(lines)

    total = evidence.get("total_valid_records", 0)
    positive = evidence.get("positive_records", 0)

    percentage = evidence.get(
        "return_rate_pct",
        evidence.get("percentage")
    )

    return "\n".join([
        f"## {get_text(language, 'return_rate')}",
        "",
        f"**{get_text(language, 'total_records')}:** "
        f"{format_number(total)}",
        "",
        f"**{get_text(language, 'positive_records')}:** "
        f"{format_number(positive)}",
        "",
        f"**{get_text(language, 'return_rate')}:** "
        f"{format_percentage(percentage)}",
    ])


# =========================================================
# TREND
# =========================================================

def format_trend_answer(evidence, language="en"):
    metrics = evidence.get("metrics", [])
    results = evidence.get("results", [])

    if not results:
        return get_text(language, "no_results")

    granularity = evidence.get(
        "granularity",
        "month"
    )

    lines = [
        f"## {get_text(language, 'trend')}",
        "",
        f"**{get_text(language, 'time_granularity')}:** "
        f"{granularity}",
        "",
        f"### {get_text(language, 'group_result')}",
        "",
    ]

    for record in results:
        period = safe_text(
            record.get("period")
        )

        lines.append(f"**{period}**")

        for metric in metrics:
            value = record.get(metric)

            label = get_metric_label(
                metric,
                language
            )

            formatted = format_metric_value(
                value,
                metric
            )

            lines.append(
                f"- {label}: {formatted}"
            )

        lines.append("")

    analyses = evidence.get(
        "metric_analysis",
        {}
    )

    for metric in metrics:
        analysis = analyses.get(metric)

        if not analysis:
            continue

        label = get_metric_label(
            metric,
            language
        )

        lines.extend([
            f"### {label}",
            "",
        ])

        highest = analysis.get("highest", {})
        lowest = analysis.get("lowest", {})

        highest_periods = ", ".join(
            safe_text(item)
            for item in highest.get("periods", [])
        )

        lowest_periods = ", ".join(
            safe_text(item)
            for item in lowest.get("periods", [])
        )

        lines.append(
            f"- {get_text(language, 'highest')}: "
            f"{format_metric_value(highest.get('value'), metric)} "
            f"({highest_periods})"
        )

        lines.append(
            f"- {get_text(language, 'lowest')}: "
            f"{format_metric_value(lowest.get('value'), metric)} "
            f"({lowest_periods})"
        )

        changes = analysis.get(
            "period_changes",
            []
        )

        if changes:
            lines.extend([
                "",
                f"**{get_text(language, 'changes')}:**",
            ])

            for change in changes:
                direction = change.get("direction")

                direction_label = get_text(
                    language,
                    direction
                ) if direction in (
                    "increase",
                    "decrease",
                    "unchanged"
                ) else safe_text(direction)

                percentage = change.get(
                    "percentage_change"
                )

                lines.append(
                    f"- {change.get('from_period')} → "
                    f"{change.get('to_period')}: "
                    f"{direction_label} "
                    f"({format_percentage(percentage)})"
                )

        lines.append("")

    return "\n".join(lines)


# =========================================================
# FILTER
# =========================================================

def format_filter_answer(evidence, language="en"):
    results = evidence.get("results", [])

    total_matches = evidence.get(
        "total_matches",
        len(results)
    )

    displayed = evidence.get(
        "displayed_records",
        len(results)
    )

    lines = [
        f"## {get_text(language, 'analysis_result')}",
        "",
        f"**{get_text(language, 'total_matches')}:** "
        f"{format_number(total_matches)}",
        "",
        f"**{get_text(language, 'displayed_records')}:** "
        f"{format_number(displayed)}",
        "",
        f"### {get_text(language, 'records')}",
        "",
    ]

    if not results:
        lines.append(
            get_text(language, "no_results")
        )

        return "\n".join(lines)

    for index, record in enumerate(
        results[:MAX_FILTER_DISPLAY],
        start=1
    ):
        lines.append(f"**{index}.**")

        for key, value in record.items():
            lines.append(
                f"- {get_column_label(key, language)}: "
                f"{safe_text(value)}"
            )

        lines.append("")

    return "\n".join(lines)


# =========================================================
# DETERMINISTIC ANSWER ROUTER
# =========================================================

def generate_deterministic_answer(
    question,
    evidence,
    query=None
):
    """Format analytical evidence without using an LLM."""

    language = resolve_output_language(
        question
    )

    operation = evidence.get("operation")

    formatters = {
        "groupby": format_groupby_answer,
        "count": format_count_answer,
        "sum": format_statistic_answer,
        "mean": format_statistic_answer,
        "median": format_statistic_answer,
        "min": format_statistic_answer,
        "max": format_statistic_answer,
        "return_rate": format_return_rate_answer,
        "trend_by_date": format_trend_answer,
        "filter": format_filter_answer,
    }

    formatter = formatters.get(operation)

    if formatter:
        return formatter(
            evidence,
            language
        )

    return None


# =========================================================
# OLLAMA FALLBACK
# =========================================================


def _generate_llm_answer_with_status(
    question,
    evidence,
    model=None
):
    """
    Generate an answer using an explicitly selected Ollama model.

    Returns:
        (answer, succeeded, error)

    Numerical calculations are never delegated to the LLM.
    """

    language = resolve_output_language(question)

    language_name = (
        "Persian"
        if language == "fa"
        else "English"
    )

    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    # An explicit None means that no model is active.
    if model is None:

        if language == "fa":
            message = (
                "برای تکمیل پاسخ به مدل زبانی نیاز است، "
                "اما در حال حاضر هیچ مدل فعالی انتخاب نشده است. "
                "از بخش AI Model Manager یک مدل را انتخاب و فعال کنید."
            )
        else:
            message = (
                "This question requires an LLM-generated explanation, "
                "but no model is currently active. "
                "Select and activate a model in AI Model Manager."
            )

        return (
            message + "\n\n" + evidence_json,
            False,
            "No active Ollama model.",
        )

    prompt = f"""
You are a professional general-purpose data analyst.

Answer the user's question using ONLY the supplied analytical evidence.

Rules:
1. Respond in {language_name}.
2. Never invent data or numerical values.
3. Never recalculate numerical results.
4. Preserve all numerical values exactly.
5. Preserve original category names and labels.
6. Clearly distinguish facts from interpretations.
7. If evidence is insufficient, explicitly say so.
8. Do not introduce information absent from the evidence.

USER QUESTION:
{question}

ANALYTICAL EVIDENCE:
{evidence_json}

Provide a clear and concise answer.
"""

    try:

        answer = ask_llm(
            prompt=prompt,
            model=model
        )

        if isinstance(answer, str) and answer.strip():

            return (
                answer.strip(),
                True,
                None,
            )

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    except Exception as error:

        error_message = str(error)

        print(
            f"Ollama answer generation failed: {error_message}"
        )

        if language == "fa":
            fallback_message = (
                "ارتباط با مدل زبانی ناموفق بود. "
                "اطلاعات تحلیلی محاسبه‌شده در ادامه آمده است."
            )
        else:
            fallback_message = (
                "LLM generation failed. "
                "The calculated analytical evidence is provided below."
            )

        return (
            fallback_message + "\n\n" + evidence_json,
            False,
            error_message,
        )


def generate_llm_answer(
    question,
    evidence,
    model=_DEFAULT_MODEL
):
    """
    Generate a natural-language answer using Ollama.

    If model is omitted, the configured default model is used
    for backward compatibility.

    If model=None is explicitly supplied, no model is active.
    """

    selected_model = (
        OLLAMA_MODEL
        if model is _DEFAULT_MODEL
        else model
    )

    answer, _, _ = _generate_llm_answer_with_status(
        question,
        evidence,
        model=selected_model
    )

    return answer


# MAIN ANSWER GENERATOR
# =========================================================


def generate_answer(
    question,
    evidence,
    query=None,
    model=_DEFAULT_MODEL
):
    """Select deterministic formatting or LLM fallback."""

    answer = generate_deterministic_answer(
        question,
        evidence,
        query
    )

    if answer is not None:
        return answer

    selected_model = (
        OLLAMA_MODEL
        if model is _DEFAULT_MODEL
        else model
    )

    return generate_llm_answer(
        question,
        evidence,
        model=selected_model
    )


# CHART REQUEST DETECTION
# =========================================================

def is_chart_request(question):
    """Detect whether the user explicitly requests a chart."""

    keywords = [
        "chart",
        "plot",
        "graph",
        "visualize",
        "visualization",
        "diagram",
        "نمودار",
        "ترسیم",
        "رسم",
        "گراف",
    ]

    normalized = str(question).lower()

    return any(
        keyword in normalized
        for keyword in keywords
    )


# =========================================================
# CHART GENERATION
# =========================================================

def generate_requested_charts(
    question,
    evidence
):
    """Generate charts for supported analytical operations."""

    if not is_chart_request(question):
        return []

    if evidence.get("operation") != "trend_by_date":
        return []

    try:
        chart_result = generate_trend_chart(
            evidence
        )

        if chart_result is None:
            return []

        if isinstance(chart_result, list):
            return chart_result

        return [chart_result]

    except Exception as error:
        print(
            f"Chart generation failed: {error}"
        )

        return []


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_analysis_input(df, question):
    """Validate dataset and question before processing."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if not isinstance(question, str):
        raise TypeError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    return question


# =========================================================
# QUERY PIPELINE
# =========================================================

def build_analysis_evidence(df, question):
    """Parse the question and execute the analytical query."""

    query = parse_question(
        question,
        df
    )

    if not isinstance(query, dict):
        raise ValueError(
            "Question parser returned an invalid query."
        )

    evidence = execute_query(
        df,
        query
    )

    if not isinstance(evidence, dict):
        raise ValueError(
            "Query engine returned an invalid result."
        )

    return query, evidence


# =========================================================
# MAIN ASSISTANT PIPELINE
# =========================================================


def _format_eda_answer(question, evidence):
    """
    Produce a deterministic exploratory analysis report.

    All numerical values come directly from EDA evidence.
    No calculations or data modifications are performed here.
    """

    language = resolve_output_language(question)

    dataset = evidence.get("dataset", {})
    numeric = evidence.get("numeric", {})
    categorical = evidence.get("categorical", {})
    warnings = evidence.get("quality_warnings", [])
    outliers = evidence.get("outliers", {})
    dates = evidence.get("dates", {})

    if language == "fa":
        lines = [
            "# ????? ????? ??????? ????",
            "",
            "## ?. ???? ??? ????",
            f"- ????? ???????: {dataset.get('rows', 0)}",
            f"- ????? ???????: {dataset.get('columns', 0)}",
            f"- ????? ?? ???????: {dataset.get('cells', 0)}",
            f"- ???????? ??????: {dataset.get('duplicate_rows', 0)}",
            f"- ???????? ????: {dataset.get('missing_cells', 0)}",
            f"- ???? ???????? ????: {dataset.get('missing_percentage', 0)}?",
        ]

        if dataset.get("column_names"):
            lines.extend([
                "",
                "???????? ?????:",
                ", ".join(map(str, dataset["column_names"])),
            ])

        lines.extend(["", "## ?. ???????? ????"])

        if numeric:
            for column, stats in numeric.items():
                lines.append(f"### {column}")
                lines.append(
                    f"- ????? ?????? ?????: {stats.get('count')}"
                )
                lines.append(
                    f"- ???????: {stats.get('mean')}"
                )
                lines.append(
                    f"- ?????: {stats.get('median')}"
                )
                lines.append(
                    f"- ?????: {stats.get('min')}"
                )
                lines.append(
                    f"- ??????: {stats.get('max')}"
                )
                lines.append(
                    f"- ?????? ????: {stats.get('negative_count')}"
                )
        else:
            lines.append("???? ???? ???? ????? ??????? ???.")

        lines.extend(["", "## ?. ???????? ???????"])

        if categorical:
            for column, stats in categorical.items():
                lines.append(
                    f"- **{column}**: "
                    f"{stats.get('unique_count')} ????? ???? "
                    f"?? {stats.get('count')} ????? ?????"
                )

                for item in stats.get("top_values", [])[:3]:
                    lines.append(
                        f"  - {item.get('value')}: "
                        f"{item.get('count')} ???? "
                        f"({item.get('percentage')}?)"
                    )
        else:
            lines.append("???? ??????? ???? ????? ??????? ???.")

        lines.extend(["", "## ?. ???????? ????? ????"])

        if warnings:
            for warning in warnings:
                lines.append(
                    f"- **{warning.get('type', 'warning')}**"
                    f" | ????: {warning.get('column', '?????')}"
                    f" | ?????: {warning.get('count', '??????')}"
                    f" | {warning.get('message', '')}"
                )
        else:
            lines.append("????? ????? ?? ????????? ????????? ???? ???.")

        lines.extend(["", "## ?. ????? ???????? ???"])

        if outliers:
            for column, stats in outliers.items():
                lines.append(
                    f"- {column}: {stats.get('count', 0)} "
                    "????? ?????? ??? ??????? ?????"
                )
        else:
            lines.append("????? ?? ????? ???????? ???????? ??? ??? ???? ???.")

        lines.extend(["", "## ?. ????? ????????"])

        if dates:
            for column, stats in dates.items():
                lines.append(f"- **{column}**: {stats}")
        else:
            lines.append("???? ?????? ???? ????? ??????? ???.")

        lines.extend([
            "",
            "## ??????????? ?????",
            "- ?????? ??? ??????? ??? ??????.",
            "- ?????? ???? ???? ????? ????? ???????? ??????? ???? ????????.",
            "- ??????? ?? ???? ????? ??? ? ?????? ????.",
            "- ??? ????? ??? ?? ????? ???? ???.",
        ])

    else:
        lines = [
            "# Exploratory Data Analysis Report",
            "",
            "## 1. Dataset Overview",
            f"- Rows: {dataset.get('rows', 0)}",
            f"- Columns: {dataset.get('columns', 0)}",
            f"- Total cells: {dataset.get('cells', 0)}",
            f"- Duplicate rows: {dataset.get('duplicate_rows', 0)}",
            f"- Missing cells: {dataset.get('missing_cells', 0)}",
            f"- Missing percentage: {dataset.get('missing_percentage', 0)}%",
        ]

        if dataset.get("column_names"):
            lines.extend([
                "",
                "Columns:",
                ", ".join(map(str, dataset["column_names"])),
            ])

        lines.extend(["", "## 2. Numeric Statistics"])

        if numeric:
            for column, stats in numeric.items():
                lines.append(f"### {column}")
                lines.append(f"- Valid values: {stats.get('count')}")
                lines.append(f"- Mean: {stats.get('mean')}")
                lines.append(f"- Median: {stats.get('median')}")
                lines.append(f"- Minimum: {stats.get('min')}")
                lines.append(f"- Maximum: {stats.get('max')}")
                lines.append(
                    f"- Negative values: {stats.get('negative_count')}"
                )
        else:
            lines.append("No numeric columns were identified.")

        lines.extend(["", "## 3. Categorical Columns"])

        if categorical:
            for column, stats in categorical.items():
                lines.append(
                    f"- **{column}**: {stats.get('unique_count')} "
                    f"unique values from {stats.get('count')} valid values"
                )

                for item in stats.get("top_values", [])[:3]:
                    lines.append(
                        f"  - {item.get('value')}: "
                        f"{item.get('count')} "
                        f"({item.get('percentage')}%)"
                    )
        else:
            lines.append("No categorical columns were identified.")

        lines.extend(["", "## 4. Data Quality Warnings"])

        if warnings:
            for warning in warnings:
                lines.append(
                    f"- **{warning.get('type', 'warning')}**"
                    f" | Column: {warning.get('column', 'General')}"
                    f" | Count: {warning.get('count', 'Unknown')}"
                    f" | {warning.get('message', '')}"
                )
        else:
            lines.append("No warnings were identified by the performed checks.")

        lines.extend(["", "## 5. Potential Outliers"])

        if outliers:
            for column, stats in outliers.items():
                lines.append(
                    f"- {column}: {stats.get('count', 0)} "
                    "potential outliers requiring review"
                )
        else:
            lines.append("No potential outliers were reported.")

        lines.extend(["", "## 6. Date Analysis"])

        if dates:
            for column, stats in dates.items():
                lines.append(f"- **{column}**: {stats}")
        else:
            lines.append("No date columns were identified.")

        lines.extend([
            "",
            "## Analysis Limitations",
            "- Outliers are screening signals, not confirmed errors.",
            "- Negative values are not automatically invalid.",
            "- Correlation does not establish causation.",
            "- No rows or values were modified or removed.",
        ])

    return "\n".join(lines)


def answer_question(
    df,
    question,
    model=_DEFAULT_MODEL
):
    """
    Run the complete general-purpose analysis pipeline.

    Exploratory requests:
        Intent classifier -> EDA engine -> evidence -> LLM interpretation

    Analytical requests:
        Intent classifier -> query parser -> query engine -> formatter

    Numerical calculations remain deterministic.
    """

    started_at = time.perf_counter()

    question = validate_analysis_input(
        df,
        question
    )

    # Preserve existing model-selection behavior.
    selected_model = (
        OLLAMA_MODEL
        if model is _DEFAULT_MODEL
        else model
    )

    if isinstance(selected_model, str):
        selected_model = selected_model.strip() or None

    # -----------------------------------------------------
    # 1. Classify user intent
    # -----------------------------------------------------

    classification = classify_intent(
        question=question,
        df=df,
        model=selected_model,
    )

    intent = classification.get("intent")

    # -----------------------------------------------------
    # 2. Exploratory analysis route
    # -----------------------------------------------------

    if intent == "exploratory_analysis":

        evidence = analyze_dataset(df)

        llm_evidence = build_llm_evidence(
            evidence
        )

        # The original complete evidence is retained for the UI.
        # Only the compact evidence is passed to the LLM.
        deterministic_answer = _format_eda_answer(
            question,
            evidence
        )

        llm_attempted = selected_model is not None
        llm_succeeded = False
        llm_error = None

        if llm_attempted:

            answer, llm_succeeded, llm_error = (
                _generate_llm_answer_with_status(
                    question,
                    llm_evidence,
                    model=selected_model,
                )
            )

            if not llm_succeeded:
                answer = deterministic_answer

        else:
            answer = deterministic_answer

        query = {
            "intent": "exploratory_analysis",
            "classifier": classification.get("classifier"),
            "confidence": classification.get("confidence"),
        }

        charts = []

        operation = "exploratory_analysis"

    # -----------------------------------------------------
    # 3. Standard analytical query route
    # -----------------------------------------------------

    else:

        query, evidence = build_analysis_evidence(
            df,
            question
        )

        deterministic_answer = generate_deterministic_answer(
            question,
            evidence,
            query
        )

        llm_attempted = False
        llm_succeeded = False
        llm_error = None

        if deterministic_answer is not None:

            answer = deterministic_answer

        else:

            llm_attempted = selected_model is not None

            answer, llm_succeeded, llm_error = (
                _generate_llm_answer_with_status(
                    question,
                    evidence,
                    model=selected_model,
                )
            )

        charts = generate_requested_charts(
            question,
            evidence
        )

        operation = evidence.get("operation")

    # -----------------------------------------------------
    # 4. Preserve the existing output contract
    # -----------------------------------------------------

    execution_time = (
        time.perf_counter() - started_at
    )

    return {
        "question": question,
        "language": resolve_output_language(question),
        "query": query,
        "evidence": evidence,
        "answer": answer,
        "charts": charts,

        "model": selected_model,
        "selected_model": selected_model,

        "llm_used": llm_succeeded,
        "llm_attempted": llm_attempted,
        "llm_succeeded": llm_succeeded,
        "llm_error": llm_error,

        "execution_time": round(execution_time, 4),
        "operation": operation,
    }
