
"""
General-Purpose Schema Detector

Responsibilities:
- Detect technical data types.
- Identify semantic column roles.
- Distinguish identifiers from numeric measures.
- Detect dates and Boolean columns.
- Support Persian and English column names.
"""

import re
import pandas as pd


DATE_NAME_HINTS = [
    "date", "time", "datetime", "timestamp",
    "تاریخ", "زمان"
]

ID_NAME_HINTS = [
    "id", "code", "uuid", "key",
    "شناسه", "کد"
]

ID_EXACT_NAMES = {
    "id", "uuid", "guid",
    "شناسه"
}

BOOLEAN_NAME_HINTS = [
    "is_", "has_", "was_", "can_",
    "active", "enabled", "disabled",
    "returned", "approved", "verified",
    "deleted", "available", "married",
    "flag", "status",
    "فعال", "تأیید", "تایید",
    "مرجوع", "وضعیت"
]

CATEGORICAL_NAME_HINTS = [
    "gender", "sex", "department",
    "category", "type", "city",
    "country", "region", "status",
    "payment_method", "product",
    "group", "level",
    "جنسیت", "دپارتمان", "بخش",
    "دسته", "نوع", "شهر", "استان",
    "کشور", "گروه", "وضعیت"
]

BOOLEAN_VALUES = {
    "yes", "no",
    "true", "false",
    "1", "0",
    "بله", "خیر",
    "درست", "نادرست",
    "فعال", "غیرفعال"
}


def normalize_name(value):
    """Normalize a column name."""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_value(value):
    """Normalize a value for comparison."""

    return str(value).strip().lower()


def contains_hint(name, hints):
    """Check whether a name contains a semantic hint."""

    normalized = normalize_name(name)

    for hint in hints:
        hint_normalized = normalize_name(hint)

        if normalized == hint_normalized:
            return True

        if (
            "_" in hint_normalized
            and hint_normalized in normalized
        ):
            return True

        if re.search(
            rf"(^|_){re.escape(hint_normalized)}($|_)",
            normalized
        ):
            return True

    return False


def is_date_column(series, column_name):
    """Detect date columns using names and values."""

    if not isinstance(series, pd.Series):
        return False

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    name_hint = contains_hint(
        column_name,
        DATE_NAME_HINTS
    )

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Numeric values should not be inferred as dates
    # unless the column has an explicit date name.
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = non_null.astype(str).head(100)

    date_pattern = sample.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$"
    )

    pattern_ratio = float(date_pattern.mean())

    if not name_hint and pattern_ratio < 0.7:
        return False

    parsed = pd.to_datetime(
        non_null,
        errors="coerce",
        format="mixed"
    )

    parsed_ratio = parsed.notna().sum() / len(non_null)

    threshold = 0.7 if name_hint else 0.85

    return parsed_ratio >= threshold


def is_identifier_column(series, column_name):
    """Detect identifier columns conservatively."""

    name = normalize_name(column_name)

    if name in ID_EXACT_NAMES:
        return True

    if contains_hint(name, ID_NAME_HINTS):
        return True

    if is_date_column(series, column_name):
        return False

    # Do not infer numeric measures as identifiers
    # only because their values are mostly unique.
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()

    if len(non_null) < 5:
        return False

    unique_ratio = non_null.nunique() / len(non_null)

    # High uniqueness can be supporting evidence
    # for textual identifier columns.
    return unique_ratio >= 0.98


def is_boolean_column(series, column_name=None):
    """Detect genuine Boolean columns."""

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    values = {
        normalize_value(value)
        for value in non_null.unique()
    }

    if len(values) != 2:
        return False

    name = normalize_name(column_name or "")

    # Explicit categorical names take priority.
    if contains_hint(name, CATEGORICAL_NAME_HINTS):
        return False

    # Recognize known Boolean value pairs.
    if values.issubset(BOOLEAN_VALUES):
        return True

    # Unknown binary categories should remain categorical.
    return contains_hint(name, BOOLEAN_NAME_HINTS)


def detect_data_type(series, column_name):
    """Detect technical data type."""

    if is_date_column(series, column_name):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "text"


def detect_column_role(series, column_name):
    """Determine semantic role of a column."""

    if is_date_column(series, column_name):
        return "date"

    if is_identifier_column(series, column_name):
        return "identifier"

    if is_boolean_column(series, column_name):
        return "boolean"

    if contains_hint(
        column_name,
        CATEGORICAL_NAME_HINTS
    ):
        return "categorical_dimension"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric_measure"

    unique_count = series.nunique(dropna=True)

    if unique_count <= 30:
        return "categorical_dimension"

    return "text"


def detect_schema(df):
    """
    Detect semantic roles and technical data types
    for all DataFrame columns.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df.columns.is_unique:
        raise ValueError(
            "Dataset contains duplicate column names."
        )

    schema = []

    for column in df.columns:

        series = df[column]

        data_type = detect_data_type(
            series,
            column
        )

        role = detect_column_role(
            series,
            column
        )

        schema.append({
            "column": str(column),
            "data_type": data_type,
            "semantic_role": role,
            "unique_count": int(
                series.nunique(dropna=True)
            ),
            "missing_count": int(
                series.isna().sum()
            )
        })

    return schema


def print_schema(schema):
    """Print detected schema."""

    print("\n========== DETECTED SCHEMA ==========")

    for item in schema:

        print(f"\nColumn: {item['column']}")
        print(f"Data Type: {item['data_type']}")
        print(f"Semantic Role: {item['semantic_role']}")
        print(f"Unique Values: {item['unique_count']}")
        print(f"Missing Values: {item['missing_count']}")
