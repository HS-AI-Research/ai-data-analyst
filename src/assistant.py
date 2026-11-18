
"""
General-Purpose Data Analysis Assistant

Responsibilities:
- Coordinate question parsing and query execution.
- Generate deterministic analytical answers.
- Support Persian and English responses.
- Preserve original dataset category values.
- Generate charts for supported analytical requests.
- Use Ollama only when deterministic formatting is insufficient.

All numerical calculations are performed by query_engine.py.
"""

import os
import json
import math
import re

import pandas as pd
from ollama import Client

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE,
)

from src.intent_parser import parse_question
from src.query_engine import execute_query
from src.metric_registry import get_metric
from src.chart_generator import generate_trend_chart
from src.dataset_context import build_dataset_context


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_CLIENT = Client(host=OLLAMA_HOST)

DEFAULT_OUTPUT_LANGUAGE = os.getenv(
    "OUTPUT_LANGUAGE",
    "auto"
).strip().lower()

MAX_FILTER_DISPLAY = 10


# =========================================================
# LANGUAGE
# =========================================================

def detect_language(text):
    text = str(text)

    if re.search(r"[\u0600-\u06FF]", text):
        return "fa"

    return "en"


def resolve_output_language(question):
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
    },
}


def get_text(language, key):
    language = language if language in MESSAGES else "en"

    return MESSAGES[language].get(
        key,
        key
    )


# =========================================================
# VALUE UTILITIES
# =========================================================

def is_missing(value):
    try:
        result = pd.isna(value)

        if hasattr(result, "ndim") and result.ndim > 0:
            return False

        return bool(result)

    except (TypeError, ValueError):
        return False


def safe_text(value):
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
    formatted = format_number(value)

    if formatted == "-":
        return "-"

    return f"{formatted}%"


def format_metric_value(value, metric_name=None):
    """
    Format metric values using metric definitions.

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
    if not column:
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
    if not metric_name:
        return get_text(language, "metric")

    definition = get_metric(metric_name)

    if language == "fa" and definition:
        return definition.get("label", metric_name)

    return str(metric_name).replace("_", " ").title()


def get_aggregation_label(aggregation, language="en"):
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

def generate_llm_answer(question, evidence):
    language = resolve_output_language(
        question
    )

    language_name = (
        "Persian"
        if language == "fa"
        else "English"
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

USER QUESTION:
{question}

ANALYTICAL EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, default=str)}

Provide a clear and concise answer.
"""

    try:
        response = OLLAMA_CLIENT.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": TEMPERATURE,
            },
        )

        message = response.get(
            "message",
            {}
        )

        content = message.get(
            "content",
            ""
        )

        if content:
            return content.strip()

    except Exception as error:
        print(
            f"Ollama answer generation failed: {error}"
        )

    return json.dumps(
        evidence,
        ensure_ascii=False,
        indent=2,
        default=str
    )


# =========================================================
# MAIN ANSWER GENERATOR
# =========================================================

def generate_answer(
    question,
    evidence,
    query=None
):
    answer = generate_deterministic_answer(
        question,
        evidence,
        query
    )

    if answer is not None:
        return answer

    return generate_llm_answer(
        question,
        evidence
    )


# =========================================================
# CHART REQUEST DETECTION
# =========================================================

def is_chart_request(question):
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
# MAIN ASSISTANT PIPELINE
# =========================================================

def answer_question(df, question):
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

    answer = generate_answer(
        question,
        evidence,
        query
    )

    charts = generate_requested_charts(
        question,
        evidence
    )

    return {
        "question": question,
        "language": resolve_output_language(question),
        "query": query,
        "evidence": evidence,
        "answer": answer,
        "charts": charts,
    }
