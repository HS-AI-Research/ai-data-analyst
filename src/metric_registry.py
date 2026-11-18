"""
Metric Registry
===============

Central registry for:
- Generic analytical operations
- Generic dataset metrics
- Domain-specific business metrics

All calculations are executed by query_engine.py.
This module only defines metadata, aliases,
requirements, resolution, and validation.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Iterable


# =========================================================
# 1. ANALYTICAL OPERATIONS
# =========================================================

OPERATION_REGISTRY = {

    "sum": {
        "label": "مجموع",
        "aliases": [
            "sum", "total", "summation",
            "مجموع", "جمع", "جمع کل",
        ],
        "calculation": "sum",
        "aggregation": "sum",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": True,
    },

    "average": {
        "label": "میانگین",
        "aliases": [
            "average", "mean", "avg",
            "arithmetic mean", "میانگین",
        ],
        "calculation": "mean",
        "aggregation": "mean",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": True,
    },

    "median": {
        "label": "میانه",
        "aliases": ["median", "میانه"],
        "calculation": "median",
        "aggregation": "median",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": True,
    },

    "minimum": {
        "label": "حداقل",
        "aliases": [
            "minimum", "min", "lowest", "smallest",
            "حداقل", "کمترین",
        ],
        "calculation": "min",
        "aggregation": "min",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": True,
    },

    "maximum": {
        "label": "حداکثر",
        "aliases": [
            "maximum", "max", "highest", "largest",
            "حداکثر", "بیشترین",
        ],
        "calculation": "max",
        "aggregation": "max",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": True,
    },

    "count": {
        "label": "تعداد",
        "aliases": [
            "count", "number of",
            "تعداد", "شمارش",
        ],
        "calculation": "count",
        "aggregation": "count",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": False,
    },

    "unique_count": {
        "label": "تعداد مقادیر یکتا",
        "aliases": [
            "unique count",
            "distinct count",
            "number of unique values",
            "number of distinct values",
            "تعداد مقادیر یکتا",
            "تعداد موارد غیرتکراری",
            "تعداد متمایز",
        ],
        "calculation": "nunique",
        "aggregation": "nunique",
        "kind": "operation",
        "domain": "generic",
        "requires_numeric_column": False,
    },
}


# =========================================================
# 2. GENERIC METRICS
# =========================================================

GENERIC_METRIC_REGISTRY = {

    "row_count": {
        "label": "تعداد رکوردها",
        "aliases": [
            "row count",
            "record count",
            "number of rows",
            "total rows",
            "number of records",
            "تعداد رکورد",
            "تعداد رکوردها",
            "تعداد ردیف",
            "تعداد ردیف ها",
            "تعداد سطر",
        ],
        "required_columns": [],
        "optional_columns": [],
        "calculation": "row_count",
        "aggregation": None,
        "kind": "metric",
        "domain": "generic",
        "unit": "records",
    },
}


# =========================================================
# 3. DOMAIN METRICS
# =========================================================

DOMAIN_METRIC_REGISTRY = {

    # -----------------------------------------------------
    # SALES
    # -----------------------------------------------------

    "revenue": {
        "label": "درآمد فروش",
        "aliases": [
            "revenue",
            "sales revenue",
            "total revenue",
            "sales amount",
            "total sales",
            "sales value",
            "درآمد",
            "فروش",
            "درآمد فروش",
            "مبلغ فروش",
            "مجموع فروش",
            "ارزش فروش",
        ],
        "required_columns": [
            "quantity",
            "unit_price_usd",
        ],
        "optional_columns": [
            "discount_pct",
        ],
        "calculation": "sales_revenue",
        "aggregation": "sum",
        "kind": "metric",
        "domain": "sales",
        "unit": "USD",
    },

    "quantity": {
        "label": "مجموع تعداد اقلام فروخته‌شده",
        "aliases": [
            "quantity",
            "units sold",
            "items sold",
            "total quantity",
            "sold quantity",
            "تعداد اقلام فروخته شده",
            "مجموع تعداد کالا",
            "تعداد کالاهای فروخته شده",
            "مقدار فروش",
        ],
        "required_columns": ["quantity"],
        "optional_columns": [],
        "calculation": "column_sum",
        "aggregation": "sum",
        "kind": "metric",
        "domain": "sales",
        "unit": "units",
    },

    "orders": {
        "label": "تعداد سفارش‌ها",
        "aliases": [
            "orders",
            "order count",
            "number of orders",
            "total orders",
            "distinct orders",
            "تعداد سفارش",
            "تعداد سفارش ها",
            "تعداد سفارش‌ها",
        ],
        "required_columns": ["order_id"],
        "optional_columns": [],
        "calculation": "unique_count",
        "aggregation": "nunique",
        "kind": "metric",
        "domain": "sales",
        "unit": "orders",
    },

    "return_rate": {
        "label": "نرخ مرجوعی",
        "aliases": [
            "return rate",
            "returns percentage",
            "percentage of returned orders",
            "returned rate",
            "نرخ مرجوعی",
            "درصد مرجوعی",
            "درصد سفارش های مرجوعی",
            "درصد سفارش‌های مرجوعی",
        ],
        "required_columns": ["returned"],
        "optional_columns": [],
        "calculation": "boolean_percentage",
        "aggregation": None,
        "kind": "metric",
        "domain": "sales",
        "unit": "percent",
    },

    "average_price": {
        "label": "میانگین قیمت واحد",
        "aliases": [
            "average price",
            "mean price",
            "unit price average",
            "average unit price",
            "میانگین قیمت",
            "میانگین قیمت واحد",
        ],
        "required_columns": ["unit_price_usd"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "sales",
        "unit": "USD",
    },

    "average_rating": {
        "label": "میانگین امتیاز مشتریان",
        "aliases": [
            "average rating",
            "mean rating",
            "customer rating",
            "customer satisfaction rating",
            "میانگین امتیاز",
            "میانگین امتیاز مشتریان",
            "امتیاز مشتریان",
        ],
        "required_columns": ["customer_rating"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "sales",
        "unit": "rating",
    },

    "discount": {
        "label": "میانگین درصد تخفیف",
        "aliases": [
            "discount",
            "average discount",
            "discount percentage",
            "discount rate",
            "میانگین تخفیف",
            "درصد تخفیف",
            "میانگین درصد تخفیف",
        ],
        "required_columns": ["discount_pct"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "sales",
        "unit": "percent",
    },

    # -----------------------------------------------------
    # HUMAN RESOURCES
    # -----------------------------------------------------

    "employee_count": {
        "label": "تعداد کارکنان",
        "aliases": [
            "employee count",
            "number of employees",
            "total employees",
            "employees",
            "staff count",
            "personnel count",
            "تعداد کارکنان",
            "تعداد کارمندان",
            "تعداد پرسنل",
            "شمار کارکنان",
        ],
        "required_columns": ["employee_id"],
        "optional_columns": [],
        "calculation": "unique_count",
        "aggregation": "nunique",
        "kind": "metric",
        "domain": "hr",
        "unit": "employees",
    },

    "total_salary": {
        "label": "مجموع حقوق",
        "aliases": [
            "total salary",
            "sum of salaries",
            "total salaries",
            "salary total",
            "مجموع حقوق",
            "جمع حقوق",
            "کل حقوق",
            "مجموع دستمزد",
        ],
        "required_columns": ["salary"],
        "optional_columns": [],
        "calculation": "column_sum",
        "aggregation": "sum",
        "kind": "metric",
        "domain": "hr",
        "unit": "currency",
    },

    "average_salary": {
        "label": "میانگین حقوق",
        "aliases": [
            "average salary",
            "mean salary",
            "avg salary",
            "average employee salary",
            "میانگین حقوق",
            "میانگین دستمزد",
            "متوسط حقوق",
            "متوسط دستمزد",
        ],
        "required_columns": ["salary"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "hr",
        "unit": "currency",
    },

    "average_age": {
        "label": "میانگین سن کارکنان",
        "aliases": [
            "average age",
            "mean age",
            "employee average age",
            "میانگین سن",
            "میانگین سن کارکنان",
            "متوسط سن",
        ],
        "required_columns": ["age"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "hr",
        "unit": "years",
    },

    "average_experience": {
        "label": "میانگین سابقه کاری",
        "aliases": [
            "average experience",
            "mean experience",
            "average work experience",
            "average years of experience",
            "میانگین سابقه",
            "میانگین سابقه کاری",
            "متوسط سابقه کار",
            "میانگین سنوات کاری",
        ],
        "required_columns": ["experience_years"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "hr",
        "unit": "years",
    },

    "average_performance": {
        "label": "میانگین امتیاز عملکرد",
        "aliases": [
            "average performance",
            "mean performance",
            "average performance score",
            "performance average",
            "میانگین عملکرد",
            "میانگین امتیاز عملکرد",
            "متوسط عملکرد",
            "میانگین ارزیابی عملکرد",
        ],
        "required_columns": ["performance_score"],
        "optional_columns": [],
        "calculation": "column_mean",
        "aggregation": "mean",
        "kind": "metric",
        "domain": "hr",
        "unit": "score",
    },
}


# =========================================================
# 4. UNIFIED REGISTRY
# =========================================================

def build_metric_registry():
    """
    Build a unified registry.

    Raises an error if duplicate metric names exist
    across registry categories.
    """

    registries = [
        OPERATION_REGISTRY,
        GENERIC_METRIC_REGISTRY,
        DOMAIN_METRIC_REGISTRY,
    ]

    result = {}

    for registry in registries:
        for name, definition in registry.items():

            if name in result:
                raise ValueError(
                    f"Duplicate registry entry: {name}"
                )

            result[name] = definition

    return result


METRIC_REGISTRY = build_metric_registry()


# =========================================================
# 5. TEXT NORMALIZATION
# =========================================================

def normalize_metric_text(value: Any) -> str:
    """Normalize Persian and English text."""

    if value is None:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "\u200c": " ",
        "\u200f": "",
        "\ufeff": "",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"

    for index in range(10):
        value = value.replace(
            persian_digits[index],
            str(index)
        )
        value = value.replace(
            arabic_digits[index],
            str(index)
        )

    value = value.lower().strip()

    value = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        value
    )

    return " ".join(value.split())


def normalize_column_name(value: Any) -> str:
    """Normalize a column name for comparison."""

    value = normalize_metric_text(value)

    value = re.sub(
        r"[\s\-]+",
        "_",
        value
    )

    value = re.sub(
        r"_+",
        "_",
        value
    )

    return value.strip("_")


def contains_metric_phrase(
    text: str,
    phrase: str
) -> bool:
    """Match a complete phrase, not a partial word."""

    text = normalize_metric_text(text)
    phrase = normalize_metric_text(phrase)

    if not text or not phrase:
        return False

    return f" {phrase} " in f" {text} "


# =========================================================
# 6. REGISTRY ACCESS
# =========================================================

def get_metric(
    metric_name: str
) -> Optional[Dict[str, Any]]:
    """Return a metric or operation definition."""

    if not metric_name:
        return None

    return METRIC_REGISTRY.get(metric_name)


def get_operation(
    operation_name: str
) -> Optional[Dict[str, Any]]:
    """Return an analytical operation definition."""

    if not operation_name:
        return None

    return OPERATION_REGISTRY.get(operation_name)


def get_metric_aliases(
    metric_name: str
) -> List[str]:
    """Return the canonical name, label, and aliases."""

    definition = get_metric(metric_name)

    if not definition:
        return []

    aliases = [
        metric_name,
        definition.get("label", ""),
        *definition.get("aliases", []),
    ]

    result = []
    seen = set()

    for alias in aliases:

        if not alias:
            continue

        normalized = normalize_metric_text(alias)

        if normalized and normalized not in seen:
            result.append(alias)
            seen.add(normalized)

    return result


def get_metric_requirements(
    metric_name: str
) -> Dict[str, List[str]]:
    """Return required and optional columns."""

    definition = get_metric(metric_name)

    if not definition:
        return {
            "required_columns": [],
            "optional_columns": [],
        }

    return {
        "required_columns": list(
            definition.get("required_columns", [])
        ),
        "optional_columns": list(
            definition.get("optional_columns", [])
        ),
    }


# =========================================================
# 7. METRIC CLASSIFICATION
# =========================================================

def is_operation(metric_name: str) -> bool:
    definition = get_metric(metric_name)

    return bool(
        definition
        and definition.get("kind") == "operation"
    )


def is_analytical_metric(metric_name: str) -> bool:
    definition = get_metric(metric_name)

    return bool(
        definition
        and definition.get("kind") == "metric"
    )


def get_metrics_by_type(
    kind: str = "metric"
) -> Dict[str, Dict[str, Any]]:

    return {
        name: definition
        for name, definition in METRIC_REGISTRY.items()
        if definition.get("kind") == kind
    }


# =========================================================
# 8. METRIC RESOLUTION
# =========================================================

def resolve_metric(
    question: str
) -> Optional[str]:
    """
    Resolve a question to one metric or operation.

    The longest matching alias wins.
    """

    if not question:
        return None

    matches = []

    for metric_name in METRIC_REGISTRY:

        for alias in get_metric_aliases(metric_name):

            normalized_alias = normalize_metric_text(alias)

            if contains_metric_phrase(
                question,
                normalized_alias
            ):
                matches.append({
                    "name": metric_name,
                    "length": len(normalized_alias),
                })

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item["length"],
            item["name"],
        ),
        reverse=True
    )

    return matches[0]["name"]


def resolve_metric_candidates(
    question: str
) -> List[Dict[str, Any]]:
    """Return all metrics matching a question."""

    if not question:
        return []

    matches = []

    for metric_name, definition in METRIC_REGISTRY.items():

        matched_aliases = []

        for alias in get_metric_aliases(metric_name):

            normalized_alias = normalize_metric_text(alias)

            if contains_metric_phrase(
                question,
                normalized_alias
            ):
                matched_aliases.append(normalized_alias)

        if not matched_aliases:
            continue

        best_alias = max(
            matched_aliases,
            key=len
        )

        matches.append({
            "name": metric_name,
            "label": definition.get("label", metric_name),
            "kind": definition.get("kind", "metric"),
            "domain": definition.get("domain", "generic"),
            "matched_alias": best_alias,
            "match_length": len(best_alias),
        })

    matches.sort(
        key=lambda item: (
            item["match_length"],
            item["name"],
        ),
        reverse=True
    )

    return matches


# =========================================================
# 9. AVAILABLE METRICS
# =========================================================

def get_available_metrics(
    columns: Iterable[str],
    include_operations: bool = True,
    include_generic: bool = True,
    include_domain: bool = True,
    domain: Optional[str] = None,
) -> List[str]:
    """
    Return metrics whose required columns exist.

    Column comparison is normalized but does not
    perform semantic column mapping.

    Semantic mapping belongs to schema_detector
    and dataset_context.
    """

    column_set = {
        normalize_column_name(column)
        for column in columns
    }

    available = []

    for metric_name, definition in METRIC_REGISTRY.items():

        kind = definition.get("kind", "metric")
        metric_domain = definition.get("domain", "generic")

        if kind == "operation":

            if not include_operations:
                continue

        else:

            if metric_domain == "generic" and not include_generic:
                continue

            if metric_domain != "generic" and not include_domain:
                continue

        if domain is not None:

            if metric_domain not in {
                domain,
                "generic",
            }:
                continue

        required_columns = {
            normalize_column_name(column)
            for column in definition.get(
                "required_columns",
                []
            )
        }

        if required_columns.issubset(column_set):
            available.append(metric_name)

    return available


def get_available_operations(
    columns=None
) -> List[str]:
    """
    Return reusable operations.

    Actual column and data-type validation is performed
    by query_engine.py.
    """

    return list(OPERATION_REGISTRY.keys())


def get_available_domain_metrics(
    columns: Iterable[str],
    domain: str
) -> List[str]:

    return get_available_metrics(
        columns,
        include_operations=False,
        include_generic=False,
        include_domain=True,
        domain=domain,
    )


# =========================================================
# 10. DOMAIN ACCESS
# =========================================================

def get_metrics_by_domain(
    domain: str,
    include_operations: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Return metrics belonging to a domain."""

    return {
        name: definition
        for name, definition in METRIC_REGISTRY.items()
        if (
            definition.get("domain") == domain
            and (
                include_operations
                or definition.get("kind") != "operation"
            )
        )
    }


def get_domain_metric_names(
    domain: str
) -> List[str]:

    return [
        name
        for name, definition in DOMAIN_METRIC_REGISTRY.items()
        if definition.get("domain") == domain
    ]


def get_supported_domains() -> List[str]:

    return sorted({
        definition.get("domain", "generic")
        for definition in METRIC_REGISTRY.values()
    })


# =========================================================
# 11. REGISTRY VALIDATION
# =========================================================

def validate_metric_registry() -> Dict[str, Any]:
    """
    Validate registry structure, aliases,
    column requirements, and calculation metadata.
    """

    errors = []
    warnings = []
    alias_owners = {}

    for metric_name, definition in METRIC_REGISTRY.items():

        if not definition.get("label"):
            errors.append(
                f"{metric_name}: missing label"
            )

        if not definition.get("calculation"):
            errors.append(
                f"{metric_name}: missing calculation"
            )

        if definition.get("kind") not in {
            "metric",
            "operation",
        }:
            errors.append(
                f"{metric_name}: invalid kind"
            )

        if not definition.get("domain"):
            errors.append(
                f"{metric_name}: missing domain"
            )

        aliases = definition.get("aliases")

        if not isinstance(aliases, list):
            errors.append(
                f"{metric_name}: aliases must be a list"
            )
            aliases = []

        if not aliases:
            warnings.append(
                f"{metric_name}: no aliases defined"
            )

        required = definition.get(
            "required_columns",
            []
        )

        optional = definition.get(
            "optional_columns",
            []
        )

        if not isinstance(required, list):
            errors.append(
                f"{metric_name}: required_columns must be a list"
            )
            required = []

        if not isinstance(optional, list):
            errors.append(
                f"{metric_name}: optional_columns must be a list"
            )
            optional = []

        overlap = set(required) & set(optional)

        if overlap:
            errors.append(
                f"{metric_name}: columns cannot be both "
                f"required and optional: {sorted(overlap)}"
            )

        if (
            definition.get("kind") == "operation"
            and required
        ):
            warnings.append(
                f"{metric_name}: operation has fixed column requirements"
            )

        for alias in [metric_name, *aliases]:

            normalized = normalize_metric_text(alias)

            if not normalized:
                continue

            previous_owner = alias_owners.get(normalized)

            if previous_owner and previous_owner != metric_name:
                warnings.append(
                    f"Alias '{alias}' is shared by "
                    f"'{previous_owner}' and '{metric_name}'"
                )
            else:
                alias_owners[normalized] = metric_name

    return {
        "valid": len(errors) == 0,
        "total_entries": len(METRIC_REGISTRY),
        "operations": len(OPERATION_REGISTRY),
        "generic_metrics": len(GENERIC_METRIC_REGISTRY),
        "domain_metrics": len(DOMAIN_METRIC_REGISTRY),
        "domains": get_supported_domains(),
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# 12. REGISTRY SUMMARY
# =========================================================

def print_registry_summary():

    result = validate_metric_registry()

    print("\n========== METRIC REGISTRY ==========")

    print(f"Total Entries: {result['total_entries']}")
    print(f"Operations: {result['operations']}")
    print(f"Generic Metrics: {result['generic_metrics']}")
    print(f"Domain Metrics: {result['domain_metrics']}")
    print(f"Domains: {', '.join(result['domains'])}")
    print(f"Valid: {result['valid']}")

    if result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"- {error}")

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    print("=====================================")