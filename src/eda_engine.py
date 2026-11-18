
"""
Exploratory Data Analysis Engine
================================

Generic, deterministic EDA for arbitrary Pandas DataFrames.

Responsibilities:
- Profile dataset structure and quality.
- Detect date-like columns without changing source data.
- Summarize numeric and categorical columns.
- Screen numeric values for potential outliers.
- Detect duplicate rows and duplicate identifier values.
- Calculate correlations only with sufficient observations.
- Build compact, evidence-grounded output for an optional LLM.

Important:
- No source data is modified.
- No rows or values are automatically removed or corrected.
- Statistical outliers are screening signals, not proof of errors.
- Correlation does not imply causation.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


MIN_OUTLIER_SAMPLE = 8
MIN_CORRELATION_SAMPLE = 8
MAX_CATEGORICAL_VALUES = 10
MAX_CORRELATION_PAIRS = 30


DATE_NAME_HINTS = (
    "date", "time", "timestamp", "datetime",
    "created", "updated", "modified",
    "تاریخ", "زمان",
)

IDENTIFIER_NAME_HINTS = (
    "id", "code", "uuid", "key",
    "شناسه", "کد",
)


# =========================================================
# 1. VALIDATION AND SAFE CONVERSION
# =========================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the input DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")

    if df.columns.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate column names."
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def _normalise_name(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    return value


def _is_identifier_column(
    column: Any,
    series: pd.Series,
) -> bool:
    """
    Identify likely identifier columns.

    Name-based detection is preferred. High-cardinality
    text columns are not automatically treated as IDs.
    """

    name = _normalise_name(column)

    if (
        name == "id"
        or name.endswith("_id")
        or name.endswith("_key")
        or name.endswith("_code")
        or name in IDENTIFIER_NAME_HINTS
    ):
        return True

    return False


# =========================================================
# 2. DATE COLUMN DETECTION
# =========================================================

def _parse_date_series(
    series: pd.Series,
) -> Optional[pd.Series]:
    """
    Try parsing a column as dates.

    Numeric columns are excluded to avoid interpreting
    ordinary measurements as Unix timestamps.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        return None

    non_null = series.dropna()

    if len(non_null) < 2:
        return None

    try:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

    valid_ratio = (
        parsed.notna().sum() / len(non_null)
        if len(non_null)
        else 0
    )

    name = _normalise_name(series.name)

    has_date_hint = any(
        hint in name
        for hint in DATE_NAME_HINTS
    )

    threshold = 0.60 if has_date_hint else 0.85

    if valid_ratio >= threshold:
        return parsed

    return None


def detect_date_columns(
    df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Return detected date columns and parsed values."""

    result = {}

    for column in df.columns:
        parsed = _parse_date_series(df[column])

        if parsed is not None:
            result[str(column)] = parsed

    return result


# =========================================================
# 3. DATASET STRUCTURE
# =========================================================

def profile_structure(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    total_cells = total_rows * total_columns

    missing_by_column = {}

    for column in df.columns:
        count = int(df[column].isna().sum())

        if count:
            missing_by_column[str(column)] = {
                "count": count,
                "percentage": round(
                    count / total_rows * 100,
                    2,
                ),
            }

    return {
        "rows": total_rows,
        "columns": total_columns,
        "cells": total_cells,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": round(
            df.isna().sum().sum() / total_cells * 100,
            2,
        ) if total_cells else 0.0,
        "missing_by_column": missing_by_column,
        "column_names": [
            str(column)
            for column in df.columns
        ],
    }


# =========================================================
# 4. NUMERIC ANALYSIS
# =========================================================

def profile_numeric_columns(
    df: pd.DataFrame,
    date_columns: Dict[str, pd.Series],
) -> Dict[str, Dict[str, Any]]:
    """Calculate descriptive statistics for numeric measures."""

    result = {}

    for column in df.select_dtypes(
        include=[np.number]
    ).columns:

        if _is_identifier_column(column, df[column]):
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if values.empty:
            continue

        result[str(column)] = {
            "count": int(values.count()),
            "missing": int(df[column].isna().sum()),
            "mean": _safe_float(values.mean()),
            "median": _safe_float(values.median()),
            "std": _safe_float(values.std()),
            "min": _safe_float(values.min()),
            "q1": _safe_float(values.quantile(0.25)),
            "q3": _safe_float(values.quantile(0.75)),
            "max": _safe_float(values.max()),
            "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
            "positive_count": int((values > 0).sum()),
        }

    return result


# =========================================================
# 5. CATEGORICAL ANALYSIS
# =========================================================

def profile_categorical_columns(
    df: pd.DataFrame,
    date_columns: Dict[str, pd.Series],
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if str(column) in date_columns:
            continue

        if _is_identifier_column(column, series):
            continue

        non_null = series.dropna()

        if non_null.empty:
            continue

        counts = non_null.astype(str).value_counts(
            dropna=True
        )

        top_values = [
            {
                "value": str(value),
                "count": int(count),
                "percentage": round(
                    count / len(non_null) * 100,
                    2,
                ),
            }
            for value, count in counts.head(
                MAX_CATEGORICAL_VALUES
            ).items()
        ]

        result[str(column)] = {
            "count": int(len(non_null)),
            "unique_count": int(non_null.nunique()),
            "top_values": top_values,
        }

    return result


# =========================================================
# 6. DATE ANALYSIS
# =========================================================

def profile_dates(
    date_columns: Dict[str, pd.Series],
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for column, parsed in date_columns.items():

        valid = parsed.dropna()

        if valid.empty:
            continue

        result[column] = {
            "valid_count": int(valid.count()),
            "invalid_or_missing_count": int(
                parsed.isna().sum()
            ),
            "min_date": str(valid.min()),
            "max_date": str(valid.max()),
            "distinct_dates": int(valid.nunique()),
        }

    return result


# =========================================================
# 7. POTENTIAL OUTLIERS
# =========================================================

def screen_outliers(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Use the IQR rule as a screening method.

    This does not classify a value as an error.
    """

    result = {}

    for column in df.select_dtypes(
        include=[np.number]
    ).columns:

        if _is_identifier_column(column, df[column]):
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(values) < MIN_OUTLIER_SAMPLE:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        outlier_values = values[mask]

        if outlier_values.empty:
            continue

        result[str(column)] = {
            "method": "IQR",
            "sample_size": int(len(values)),
            "lower_bound": _safe_float(lower_bound),
            "upper_bound": _safe_float(upper_bound),
            "count": int(len(outlier_values)),
            "percentage": round(
                len(outlier_values) / len(values) * 100,
                2,
            ),
            "min_flagged_value": _safe_float(
                outlier_values.min()
            ),
            "max_flagged_value": _safe_float(
                outlier_values.max()
            ),
            "interpretation": (
                "Potential statistical outliers; "
                "manual review required."
            ),
        }

    return result


# =========================================================
# 8. IDENTIFIER QUALITY
# =========================================================

def inspect_identifiers(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for column in df.columns:

        if not _is_identifier_column(
            column,
            df[column],
        ):
            continue

        series = df[column]

        duplicate_mask = (
            series.notna()
            & series.duplicated(keep=False)
        )

        result[str(column)] = {
            "missing_count": int(series.isna().sum()),
            "duplicate_value_rows": int(
                duplicate_mask.sum()
            ),
            "unique_count": int(series.nunique()),
        }

    return result


# =========================================================
# 9. CORRELATION ANALYSIS
# =========================================================

def calculate_correlations(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate Pearson correlations between numeric measures.

    Identifier columns are excluded. Pairwise observations
    must meet the minimum sample threshold.
    """

    numeric_columns = [
        column
        for column in df.select_dtypes(
            include=[np.number]
        ).columns
        if not _is_identifier_column(
            column,
            df[column],
        )
    ]

    pairs = []
    skipped_pairs = 0

    for left, right in combinations(
        numeric_columns,
        2,
    ):

        pair = df[[left, right]].apply(
            pd.to_numeric,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(pair) < MIN_CORRELATION_SAMPLE:
            skipped_pairs += 1
            continue

        if pair[left].nunique() < 2:
            skipped_pairs += 1
            continue

        if pair[right].nunique() < 2:
            skipped_pairs += 1
            continue

        correlation = pair[left].corr(
            pair[right],
            method="pearson",
        )

        if pd.isna(correlation):
            skipped_pairs += 1
            continue

        pairs.append({
            "column_1": str(left),
            "column_2": str(right),
            "correlation": round(
                float(correlation),
                4,
            ),
            "sample_size": int(len(pair)),
            "strength": (
                "strong"
                if abs(correlation) >= 0.70
                else "moderate"
                if abs(correlation) >= 0.40
                else "weak"
            ),
        })

    pairs.sort(
        key=lambda item: abs(
            item["correlation"]
        ),
        reverse=True,
    )

    return {
        "method": "Pearson",
        "minimum_sample_size": MIN_CORRELATION_SAMPLE,
        "pairs": pairs[:MAX_CORRELATION_PAIRS],
        "skipped_pairs": skipped_pairs,
        "note": (
            "Correlation indicates association, not causation."
        ),
    }


# =========================================================
# 10. GENERAL QUALITY FLAGS
# =========================================================

def build_quality_warnings(
    df: pd.DataFrame,
    structure: Dict[str, Any],
    numeric: Dict[str, Dict[str, Any]],
    outliers: Dict[str, Dict[str, Any]],
    identifiers: Dict[str, Dict[str, Any]],
    dates: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:

    warnings = []

    if structure["missing_cells"] > 0:
        warnings.append({
            "type": "missing_values",
            "severity": "review",
            "count": structure["missing_cells"],
            "message": (
                "The dataset contains missing values."
            ),
        })

    if structure["duplicate_rows"] > 0:
        warnings.append({
            "type": "duplicate_rows",
            "severity": "review",
            "count": structure["duplicate_rows"],
            "message": (
                "The dataset contains duplicate rows."
            ),
        })

    for column, stats in numeric.items():

        if stats["negative_count"] > 0:
            warnings.append({
                "type": "negative_numeric_values",
                "severity": "review",
                "column": column,
                "count": stats["negative_count"],
                "message": (
                    "Negative numeric values were found. "
                    "Their validity depends on the business context."
                ),
            })

    for column, stats in outliers.items():
        warnings.append({
            "type": "potential_outliers",
            "severity": "review",
            "column": column,
            "count": stats["count"],
            "message": (
                "Potential statistical outliers require review."
            ),
        })

    for column, stats in identifiers.items():

        if stats["duplicate_value_rows"] > 0:
            warnings.append({
                "type": "duplicate_identifier_values",
                "severity": "review",
                "column": column,
                "count": stats["duplicate_value_rows"],
                "message": (
                    "Identifier values are duplicated. "
                    "Check whether uniqueness is required."
                ),
            })

    for column, stats in dates.items():

        if stats["invalid_or_missing_count"] > 0:
            warnings.append({
                "type": "invalid_or_missing_dates",
                "severity": "review",
                "column": column,
                "count": stats["invalid_or_missing_count"],
                "message": (
                    "Some values could not be parsed as dates "
                    "or are missing."
                ),
            })

    return warnings


# =========================================================
# 11. MAIN EDA PIPELINE
# =========================================================

def analyze_dataset(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run generic exploratory data analysis.

    Returns a JSON-serializable evidence dictionary.
    """

    validate_dataframe(df)

    date_columns = detect_date_columns(df)

    structure = profile_structure(df)

    numeric = profile_numeric_columns(
        df,
        date_columns,
    )

    categorical = profile_categorical_columns(
        df,
        date_columns,
    )

    dates = profile_dates(date_columns)

    outliers = screen_outliers(df)

    identifiers = inspect_identifiers(df)

    correlations = calculate_correlations(df)

    warnings = build_quality_warnings(
        df=df,
        structure=structure,
        numeric=numeric,
        outliers=outliers,
        identifiers=identifiers,
        dates=dates,
    )

    return {
        "analysis_type": "exploratory_data_analysis",
        "dataset": structure,
        "numeric": numeric,
        "categorical": categorical,
        "dates": dates,
        "outliers": outliers,
        "identifiers": identifiers,
        "correlations": correlations,
        "quality_warnings": warnings,
        "limitations": [
            "Outliers are screening signals, not confirmed errors.",
            "Negative values are not automatically invalid.",
            "Correlation does not establish causation.",
            "Date detection is heuristic and should be reviewed.",
            "No rows or values were modified or removed.",
        ],
    }


# =========================================================
# 12. REQUEST ROUTING
# =========================================================

def is_exploratory_request(question: str) -> bool:
    """
    Lightweight multilingual fallback for exploratory analysis detection.
    LLM-based classification remains the primary mechanism.
    """
    if not question:
        return False

    import re

    text = str(question).lower().strip()

    # Normalize Arabic/Persian character variants and invisible marks.
    text = (
        text.replace("\u064a", "\u06cc")
            .replace("\u0643", "\u06a9")
            .replace("\u200c", "")
            .replace("\u200e", "")
            .replace("\u200f", "")
    )

    normalized = re.sub(r"\s+", " ", text)

    english_patterns = [
        r"\bexploratory\s+data\s+analysis\b",
        r"\bperform\s+(an?\s+)?eda\b",
        r"\banaly[sz]e\s+(this|the)\s+dataset\b",
        r"\bprofile\s+(this|the)\s+dataset\b",
        r"\bdata\s+overview\b",
        r"\bbusiness\s+insights?\b",
        r"\bpotential\s+risks?\b",
        r"\bkey\s+insights?\b",
        r"\bimportant\s+insights?\b",
        r"\bwhat\s+are\s+the\s+main\s+findings\b",
    ]

    if any(re.search(pattern, normalized) for pattern in english_patterns):
        return True

    compact = re.sub(r"\s+", "", normalized)

    # Unicode-safe Persian keyword detection.
    insight_word = "\u0628\u06cc\u0646\u0634"
    risk_word = "\u0631\u06cc\u0633\u06a9"

    if insight_word in compact and risk_word in compact:
        return True

    persian_patterns = [
        "\u062a\u062d\u0644\u06cc\u0644\u0627\u06a9\u062a\u0634\u0627\u0641\u06cc",
        "\u062a\u062d\u0644\u06cc\u0644\u062c\u0627\u0645\u0639",
        "\u0628\u0631\u0631\u0633\u06cc\u062c\u0627\u0645\u0639\u062f\u0627\u062f\u0647",
        "\u06a9\u06cc\u0641\u06cc\u062a\u062f\u0627\u062f\u0647",
        "\u0648\u0636\u0639\u06cc\u062a\u062f\u0627\u062f\u0647",
        "\u0645\u0647\u0645\u062a\u0631\u06cc\u0646\u0628\u06cc\u0646\u0634",
        "\u0646\u06a9\u0627\u062a\u0645\u0647\u0645\u062f\u0627\u062f\u0647",
    ]

    return any(pattern in compact for pattern in persian_patterns)
def build_llm_evidence(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact evidence payload for an LLM.

    Does not include raw rows or categorical sample values.
    """

    numeric = {}

    for column, stats in evidence.get(
        "numeric", {}
    ).items():

        numeric[column] = {
            key: stats.get(key)
            for key in (
                "count",
                "missing",
                "mean",
                "median",
                "min",
                "q1",
                "q3",
                "max",
                "negative_count",
                "zero_count",
            )
        }

    categorical = {}

    for column, stats in evidence.get(
        "categorical", {}
    ).items():

        categorical[column] = {
            "count": stats.get("count"),
            "unique_count": stats.get("unique_count"),
            "top_values": stats.get("top_values", [])[:5],
        }

    return {
        "analysis_type": evidence.get("analysis_type"),
        "dataset": evidence.get("dataset"),
        "numeric": numeric,
        "categorical": categorical,
        "dates": evidence.get("dates"),
        "outliers": evidence.get("outliers"),
        "identifiers": evidence.get("identifiers"),
        "correlations": evidence.get("correlations"),
        "quality_warnings": evidence.get(
            "quality_warnings"
        ),
        "limitations": evidence.get("limitations"),
    }
