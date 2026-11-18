
"""
Dataset Context
===============

Dataset Intelligence Layer

Responsibilities:
    - Build a unified representation of a dataset.
    - Profile dataset structure and quality.
    - Detect column types and semantic roles.
    - Detect the most likely business domain.
    - Identify available generic and domain metrics.
    - Identify reusable analytical operations.
    - Infer general analytical capabilities from column roles.
    - Prepare structured context for the intent parser and LLM.

Important:
    This module does not perform analytical calculations.
    Actual calculations are handled by query_engine.py.

    It also does not modify original DataFrame columns or values.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_profiler import profile_dataset
from src.schema_detector import detect_schema

from src.domain_registry import (
    detect_domain,
    get_domain_metrics,
)

from src.metric_registry import (
    METRIC_REGISTRY,
    get_available_metrics,
    get_available_operations,
    get_metric,
    normalize_column_name,
    is_analytical_metric,
)


# =========================================================
# 1. GENERAL CONFIGURATION
# =========================================================

CONTEXT_VERSION = "2.0"

UNKNOWN_ROLE = "unknown"

SUPPORTED_ROLES = {
    "numeric_measure",
    "categorical_dimension",
    "date",
    "identifier",
    "boolean",
    "text",
    "unknown",
}


# =========================================================
# 2. BASIC DATASET VALIDATION
# =========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the input DataFrame.
    """

    if df is None:
        raise ValueError("Dataset cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")

    if df.columns.duplicated().any():
        duplicates = (
            df.columns[df.columns.duplicated()]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"Dataset contains duplicate column names: {duplicates}"
        )


# =========================================================
# 3. SAFE VALUE HELPERS
# =========================================================

def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to integer safely.
    """

    try:
        if pd.isna(value):
            return default

        return int(value)

    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(
    value: Any,
    default: Optional[float] = None
) -> Optional[float]:
    """
    Convert a value to float safely.
    """

    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError, OverflowError):
        return default


def safe_list(value: Any) -> List:
    """
    Return a list safely.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return []


# =========================================================
# 4. COLUMN SCHEMA HELPERS
# =========================================================

def get_schema_column_name(column_info: Dict) -> Optional[str]:
    """
    Extract the original column name from schema metadata.
    """

    return column_info.get(
        "column",
        column_info.get("name")
    )


def get_schema_role(column_info: Dict) -> str:
    """
    Extract semantic role from schema metadata.
    """

    role = column_info.get(
        "semantic_role",
        column_info.get("role", UNKNOWN_ROLE)
    )

    if not role:
        return UNKNOWN_ROLE

    return str(role)


def get_schema_data_type(column_info: Dict) -> str:
    """
    Extract detected data type from schema metadata.
    """

    data_type = column_info.get(
        "data_type",
        column_info.get(
            "dtype",
            column_info.get("type", "unknown")
        )
    )

    return str(data_type)


def build_column_catalog(
    df: pd.DataFrame,
    schema: List[Dict]
) -> List[Dict]:
    """
    Build a unified catalog of dataset columns.

    Original column names are preserved.

    The catalog combines:
        - Original DataFrame column name
        - Normalized name
        - Pandas dtype
        - Detected data type
        - Semantic role
        - Missing values
        - Unique values
        - Sample values
    """

    schema_by_name = {}

    for item in schema:

        if not isinstance(item, dict):
            continue

        name = get_schema_column_name(item)

        if name is not None:
            schema_by_name[str(name)] = item

    catalog = []

    for column in df.columns:

        original_name = str(column)

        info = schema_by_name.get(
            original_name,
            {}
        )

        series = df[column]

        missing_count = safe_int(
            series.isna().sum()
        )

        unique_count = safe_int(
            series.nunique(dropna=True)
        )

        sample_values = (
            series.dropna()
            .head(5)
            .tolist()
        )

        catalog.append({

            "column": original_name,

            "normalized_name": normalize_column_name(
                original_name
            ),

            "pandas_dtype": str(series.dtype),

            "data_type": get_schema_data_type(info),

            "semantic_role": get_schema_role(info),

            "missing_count": missing_count,

            "missing_percentage": round(
                missing_count / len(df) * 100,
                2
            ),

            "unique_count": unique_count,

            "sample_values": [
                str(value)
                for value in sample_values
            ],

            "is_numeric": (
                pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series)
            ),

            "is_datetime": (
                pd.api.types.is_datetime64_any_dtype(series)
            ),

            "is_boolean": (
                pd.api.types.is_bool_dtype(series)
            ),

        })

    return catalog


# =========================================================
# 5. COLUMN ROLE ANALYSIS
# =========================================================

def get_column_roles(
    context: Dict
) -> Dict[str, List[str]]:
    """
    Return column names grouped by semantic role.
    """

    roles = {}

    schema = context.get("schema", [])

    for column in schema:

        if not isinstance(column, dict):
            continue

        role = get_schema_role(column)

        name = get_schema_column_name(column)

        if name is None:
            continue

        roles.setdefault(role, []).append(
            str(name)
        )

    return roles


def get_columns_by_role(
    context: Dict,
    role: str
) -> List[str]:
    """
    Return columns matching a semantic role.
    """

    return get_column_roles(context).get(
        role,
        []
    )


def get_numeric_columns(context: Dict) -> List[str]:
    """
    Return numeric measure columns.
    """

    return get_columns_by_role(
        context,
        "numeric_measure"
    )


def get_categorical_columns(context: Dict) -> List[str]:
    """
    Return categorical dimension columns.
    """

    return get_columns_by_role(
        context,
        "categorical_dimension"
    )


def get_date_columns(context: Dict) -> List[str]:
    """
    Return date columns.
    """

    return get_columns_by_role(
        context,
        "date"
    )


def get_identifier_columns(context: Dict) -> List[str]:
    """
    Return identifier columns.
    """

    return get_columns_by_role(
        context,
        "identifier"
    )


def get_boolean_columns(context: Dict) -> List[str]:
    """
    Return boolean columns.
    """

    return get_columns_by_role(
        context,
        "boolean"
    )


# =========================================================
# 6. DATA QUALITY SUMMARY
# =========================================================

def build_data_quality_summary(
    df: pd.DataFrame,
    profile: Dict,
    column_catalog: List[Dict]
) -> Dict:
    """
    Build a unified data quality summary.
    """

    total_cells = len(df) * len(df.columns)

    missing_cells = safe_int(
        df.isna().sum().sum()
    )

    duplicate_rows = safe_int(
        df.duplicated().sum()
    )

    columns_with_missing_values = [
        item["column"]
        for item in column_catalog
        if item["missing_count"] > 0
    ]

    columns_with_constant_values = [
        item["column"]
        for item in column_catalog
        if item["unique_count"] <= 1
    ]

    missing_percentage = (
        round(missing_cells / total_cells * 100, 2)
        if total_cells
        else 0.0
    )

    duplicate_percentage = (
        round(duplicate_rows / len(df) * 100, 2)
        if len(df)
        else 0.0
    )

    return {

        "total_cells": total_cells,

        "missing_cells": missing_cells,

        "missing_percentage": missing_percentage,

        "duplicate_rows": duplicate_rows,

        "duplicate_percentage": duplicate_percentage,

        "columns_with_missing_values": (
            columns_with_missing_values
        ),

        "columns_with_constant_values": (
            columns_with_constant_values
        ),

        "columns_with_missing_count": len(
            columns_with_missing_values
        ),

        "columns_with_constant_count": len(
            columns_with_constant_values
        ),

        "has_missing_values": missing_cells > 0,

        "has_duplicate_rows": duplicate_rows > 0,

        "profile_available": isinstance(profile, dict),

    }


# =========================================================
# 7. METRIC DISCOVERY
# =========================================================

def discover_available_metrics(
    columns,
    domain_name: str
) -> Dict[str, List[str]]:
    """
    Discover generic and domain-specific metrics.

    Operations are intentionally excluded from semantic
    metric lists.

    Returns:
        {
            "generic": [...],
            "domain": [...],
            "all": [...]
        }
    """

    generic_metrics = get_available_metrics(
        columns,
        include_operations=False,
        include_generic=True,
        include_domain=False,
    )

    domain_metrics = get_available_metrics(
        columns,
        include_operations=False,
        include_generic=False,
        include_domain=True,
        domain=domain_name,
    )

    # Domain registry may contain metric names that are
    # not yet implemented in METRIC_REGISTRY.
    registered_domain_metrics = get_domain_metrics(
        domain_name
    )

    domain_metrics = [
        metric
        for metric in domain_metrics
        if metric in registered_domain_metrics
    ]

    all_metrics = list(dict.fromkeys(
        generic_metrics + domain_metrics
    ))

    return {

        "generic": generic_metrics,

        "domain": domain_metrics,

        "all": all_metrics,

    }


def build_metric_definitions(
    metric_names: List[str]
) -> Dict[str, Dict]:
    """
    Collect definitions for available metrics.

    Copies registry definitions to avoid modifying
    the central registry accidentally.
    """

    definitions = {}

    for metric_name in metric_names:

        definition = get_metric(metric_name)

        if definition is None:
            continue

        definitions[metric_name] = dict(
            definition
        )

    return definitions


# =========================================================
# 8. ANALYTICAL OPERATIONS
# =========================================================

def discover_available_operations() -> List[str]:
    """
    Return reusable analytical operations.

    Operations are not tied to specific business domains.

    The query engine must validate the selected column
    and data type before executing an operation.
    """

    return get_available_operations()


# =========================================================
# 9. DYNAMIC ANALYSIS CAPABILITIES
# =========================================================

def discover_analysis_capabilities(
    column_roles: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Infer general analytical capabilities from
    detected semantic column roles.

    These are capability declarations, not metric
    definitions and not guarantees of execution.
    """

    numeric_columns = column_roles.get(
        "numeric_measure",
        []
    )

    categorical_columns = column_roles.get(
        "categorical_dimension",
        []
    )

    date_columns = column_roles.get(
        "date",
        []
    )

    identifier_columns = column_roles.get(
        "identifier",
        []
    )

    boolean_columns = column_roles.get(
        "boolean",
        []
    )

    text_columns = column_roles.get(
        "text",
        []
    )

    capabilities = {

        "row_analysis": True,

        "numeric_analysis": len(numeric_columns) > 0,

        "categorical_analysis": (
            len(categorical_columns) > 0
        ),

        "time_series_analysis": (
            len(date_columns) > 0
        ),

        "group_comparison": (
            len(categorical_columns) > 0
            and len(numeric_columns) > 0
        ),

        "numeric_correlation": (
            len(numeric_columns) >= 2
        ),

        "ranking": (
            len(numeric_columns) > 0
            or len(categorical_columns) > 0
        ),

        "unique_value_analysis": (
            len(categorical_columns) > 0
            or len(identifier_columns) > 0
        ),

        "boolean_analysis": (
            len(boolean_columns) > 0
        ),

        "text_analysis": (
            len(text_columns) > 0
        ),

        "available_column_groups": {

            "numeric": numeric_columns,

            "categorical": categorical_columns,

            "date": date_columns,

            "identifier": identifier_columns,

            "boolean": boolean_columns,

            "text": text_columns,

        },

    }

    return capabilities


# =========================================================
# 10. DOMAIN CONTEXT
# =========================================================

def build_domain_context(
    domain_result: Dict
) -> Dict:
    """
    Normalize domain detection output.
    """

    domain_name = domain_result.get(
        "domain",
        "generic"
    )

    candidates = safe_list(
        domain_result.get(
            "candidate_domains",
            []
        )
    )

    return {

        "name": domain_name,

        "label": domain_result.get(
            "label",
            "عمومی"
        ),

        "confidence": safe_float(
            domain_result.get("confidence"),
            0.0
        ),

        "matched_columns": safe_list(
            domain_result.get("matched_columns")
        ),

        "candidate_domains": candidates,

        "is_generic": domain_name == "generic",

    }


# =========================================================
# 11. MAIN CONTEXT BUILDER
# =========================================================

def build_dataset_context(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None
) -> Dict:
    """
    Build a unified intelligence context for a dataset.

    This is the main entry point for dataset understanding.

    Parameters:
        df:
            Input Pandas DataFrame.

        dataset_name:
            Optional dataset name.

    Returns:
        Structured dictionary containing:

            dataset
            profile
            schema
            column_catalog
            column_roles
            data_quality
            domain
            available_metrics
            metric_categories
            metric_definitions
            available_operations
            analysis_capabilities
    """

    # -----------------------------------------------------
    # 1. Validate dataset
    # -----------------------------------------------------

    validate_dataset(df)

    # -----------------------------------------------------
    # 2. Profile dataset
    # -----------------------------------------------------

    profile = profile_dataset(df)

    if not isinstance(profile, dict):
        raise TypeError(
            "profile_dataset must return a dictionary."
        )

    profile_dataset_info = profile.get(
        "dataset",
        {}
    )

    if not isinstance(profile_dataset_info, dict):
        profile_dataset_info = {}

    # -----------------------------------------------------
    # 3. Detect schema
    # -----------------------------------------------------

    schema = detect_schema(df)

    if not isinstance(schema, list):
        raise TypeError(
            "detect_schema must return a list."
        )

    # -----------------------------------------------------
    # 4. Build column catalog
    # -----------------------------------------------------

    column_catalog = build_column_catalog(
        df,
        schema
    )

    # -----------------------------------------------------
    # 5. Detect semantic column roles
    # -----------------------------------------------------

    temporary_context = {
        "schema": schema
    }

    column_roles = get_column_roles(
        temporary_context
    )

    # -----------------------------------------------------
    # 6. Detect dataset domain
    # -----------------------------------------------------

    domain_result = detect_domain(
        df.columns,
        schema=schema
    )

    if not isinstance(domain_result, dict):
        raise TypeError(
            "detect_domain must return a dictionary."
        )

    domain_context = build_domain_context(
        domain_result
    )

    domain_name = domain_context["name"]

    # -----------------------------------------------------
    # 7. Discover available metrics
    # -----------------------------------------------------

    metric_categories = discover_available_metrics(
        df.columns,
        domain_name
    )

    available_metrics = metric_categories["all"]

    # -----------------------------------------------------
    # 8. Collect metric definitions
    # -----------------------------------------------------

    metric_definitions = build_metric_definitions(
        available_metrics
    )

    # -----------------------------------------------------
    # 9. Discover general operations
    # -----------------------------------------------------

    available_operations = (
        discover_available_operations()
    )

    # -----------------------------------------------------
    # 10. Discover dynamic capabilities
    # -----------------------------------------------------

    analysis_capabilities = (
        discover_analysis_capabilities(
            column_roles
        )
    )

    # -----------------------------------------------------
    # 11. Build data quality summary
    # -----------------------------------------------------

    data_quality = build_data_quality_summary(
        df,
        profile,
        column_catalog
    )

    # -----------------------------------------------------
    # 12. Build dataset metadata
    # -----------------------------------------------------

    dataset_info = {

        "name": dataset_name or "uploaded_dataset",

        "total_rows": len(df),

        "total_columns": len(df.columns),

        "total_cells": len(df) * len(df.columns),

        "columns": [
            str(column)
            for column in df.columns
        ],

        "duplicate_rows": safe_int(
            df.duplicated().sum()
        ),

    }

    # Preserve additional profile metadata.
    for key, value in profile_dataset_info.items():

        if key not in dataset_info:
            dataset_info[key] = value

    # -----------------------------------------------------
    # 13. Build final context
    # -----------------------------------------------------

    context = {

        "context_version": CONTEXT_VERSION,

        "dataset": dataset_info,

        "profile": profile,

        "schema": schema,

        "column_catalog": column_catalog,

        "column_roles": column_roles,

        "data_quality": data_quality,

        "domain": domain_context,

        "available_metrics": available_metrics,

        "metric_categories": metric_categories,

        "metric_definitions": metric_definitions,

        "available_operations": available_operations,

        "analysis_capabilities": analysis_capabilities,

    }

    return context


# =========================================================
# 12. CONTEXT ACCESS HELPERS
# =========================================================

def get_context_metrics(
    context: Dict,
    domain: Optional[str] = None
) -> List[str]:
    """
    Return available metrics from an existing context.

    Optionally filter by metric domain.
    """

    metrics = context.get(
        "available_metrics",
        []
    )

    if domain is None:
        return list(metrics)

    definitions = context.get(
        "metric_definitions",
        {}
    )

    return [
        metric
        for metric in metrics
        if definitions.get(
            metric,
            {}
        ).get("domain") == domain
    ]


def get_context_column(
    context: Dict,
    column_name: str
) -> Optional[Dict]:
    """
    Find a column in the column catalog.

    Supports original and normalized names.
    """

    normalized_name = normalize_column_name(
        column_name
    )

    for item in context.get("column_catalog", []):

        if item.get("column") == column_name:
            return item

        if item.get("normalized_name") == normalized_name:
            return item

    return None


def get_context_summary(
    context: Dict
) -> Dict:
    """
    Return a compact summary suitable for an LLM prompt.
    """

    dataset = context.get("dataset", {})

    domain = context.get("domain", {})

    return {

        "dataset_name": dataset.get("name"),

        "rows": dataset.get("total_rows"),

        "columns": dataset.get("total_columns"),

        "column_names": dataset.get("columns", []),

        "domain": domain.get("name"),

        "domain_label": domain.get("label"),

        "available_metrics": context.get(
            "available_metrics",
            []
        ),

        "available_operations": context.get(
            "available_operations",
            []
        ),

        "column_roles": context.get(
            "column_roles",
            {}
        ),

        "analysis_capabilities": context.get(
            "analysis_capabilities",
            {}
        ),

        "data_quality": context.get(
            "data_quality",
            {}
        ),

    }


# =========================================================
# 13. PRINT FUNCTIONS
# =========================================================

def print_dataset_context(
    context: Dict
) -> None:
    """
    Print a readable dataset intelligence summary.
    """

    dataset = context.get("dataset", {})

    domain = context.get("domain", {})

    print("\n========== DATASET CONTEXT ==========")

    print(
        f"Context Version: "
        f"{context.get('context_version')}"
    )

    print(
        f"Dataset Name: "
        f"{dataset.get('name')}"
    )

    print(
        f"Rows: "
        f"{dataset.get('total_rows')}"
    )

    print(
        f"Columns: "
        f"{dataset.get('total_columns')}"
    )

    print(
        f"Duplicate Rows: "
        f"{dataset.get('duplicate_rows')}"
    )

    # -----------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------

    print("\n========== DOMAIN ==========")

    print(
        f"Domain: {domain.get('name')}"
    )

    print(
        f"Label: {domain.get('label')}"
    )

    print(
        f"Confidence: {domain.get('confidence')}"
    )

    print(
        f"Matched Columns: "
        f"{domain.get('matched_columns')}"
    )

    # -----------------------------------------------------
    # COLUMN ROLES
    # -----------------------------------------------------

    print("\n========== COLUMN ROLES ==========")

    roles = context.get(
        "column_roles",
        {}
    )

    for role, columns in roles.items():

        print(
            f"{role}: {', '.join(columns)}"
        )

    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

    print("\n========== DATA QUALITY ==========")

    quality = context.get(
        "data_quality",
        {}
    )

    print(
        f"Missing Cells: "
        f"{quality.get('missing_cells')}"
    )

    print(
        f"Missing Percentage: "
        f"{quality.get('missing_percentage')}%"
    )

    print(
        f"Duplicate Rows: "
        f"{quality.get('duplicate_rows')}"
    )

    print(
        f"Columns With Missing Values: "
        f"{quality.get('columns_with_missing_count')}"
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    print("\n========== AVAILABLE METRICS ==========")

    definitions = context.get(
        "metric_definitions",
        {}
    )

    for metric in context.get(
        "available_metrics",
        []
    ):

        definition = definitions.get(
            metric,
            {}
        )

        print(
            f"- {metric}: "
            f"{definition.get('label', metric)}"
        )

    # -----------------------------------------------------
    # OPERATIONS
    # -----------------------------------------------------

    print("\n========== AVAILABLE OPERATIONS ==========")

    for operation in context.get(
        "available_operations",
        []
    ):

        print(f"- {operation}")

    # -----------------------------------------------------
    # CAPABILITIES
    # -----------------------------------------------------

    print("\n========== ANALYSIS CAPABILITIES ==========")

    capabilities = context.get(
        "analysis_capabilities",
        {}
    )

    for name, value in capabilities.items():

        if name == "available_column_groups":
            continue

        print(
            f"{name}: {value}"
        )

    print("\n=========================================")


# =========================================================
# 14. BACKWARD-COMPATIBLE ALIASES
# =========================================================

def build_context(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None
) -> Dict:
    """
    Alias for build_dataset_context.
    """

    return build_dataset_context(
        df,
        dataset_name
    )
