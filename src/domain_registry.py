
"""
Domain Registry
===============

Defines supported business domains and detects
the most likely domain of a dataset.

Responsibilities:
- Maintain domain metadata.
- Detect domains using column-name hints.
- Support Persian and English names.
- Expose implemented domain metrics.
- Preserve compatibility with dataset_context.py.

Actual metric definitions are maintained in
metric_registry.py.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional


# =========================================================
# 1. DOMAIN DEFINITIONS
# =========================================================

DOMAIN_REGISTRY = {

    # -----------------------------------------------------
    # SALES
    # -----------------------------------------------------

    "sales": {
        "label": "فروش",
        "aliases": [
            "sales",
            "ecommerce",
            "orders",
            "فروش",
            "فروشگاه",
            "سفارشات",
        ],
        "column_hints": [
            "order_id",
            "order_date",
            "quantity",
            "unit_price",
            "discount",
            "returned",
            "product",
            "customer",
        ],
        "metrics": [
            "revenue",
            "quantity",
            "orders",
            "return_rate",
            "average_price",
            "average_rating",
            "discount",
        ],
    },

    # -----------------------------------------------------
    # HUMAN RESOURCES
    # -----------------------------------------------------

    "hr": {
        "label": "منابع انسانی",
        "aliases": [
            "hr",
            "human resources",
            "employees",
            "personnel",
            "staff",
            "منابع انسانی",
            "کارکنان",
            "کارمندان",
            "پرسنلی",
        ],
        "column_hints": [
            "employee_id",
            "employee_name",
            "department",
            "salary",
            "hire_date",
            "termination_date",
            "job_title",
            "performance",
            "performance_score",
            "experience_years",
            "age",
        ],
        "metrics": [
            "employee_count",
            "total_salary",
            "average_salary",
            "average_age",
            "average_experience",
            "average_performance",
        ],
        "planned_metrics": [
            "turnover_rate",
            "hiring_count",
        ],
    },

    # -----------------------------------------------------
    # FINANCE
    # -----------------------------------------------------

    "finance": {
        "label": "مالی",
        "aliases": [
            "finance",
            "financial",
            "accounting",
            "مالی",
            "حسابداری",
            "مالیه",
        ],
        "column_hints": [
            "revenue",
            "income",
            "expense",
            "cost",
            "profit",
            "loss",
            "transaction",
            "account",
        ],
        "metrics": [],
        "planned_metrics": [
            "total_income",
            "total_expense",
            "profit",
            "profit_margin",
        ],
    },

    # -----------------------------------------------------
    # PROJECT MANAGEMENT
    # -----------------------------------------------------

    "project": {
        "label": "مدیریت پروژه",
        "aliases": [
            "project",
            "projects",
            "project management",
            "مدیریت پروژه",
            "پروژه",
        ],
        "column_hints": [
            "project_id",
            "task_id",
            "task_name",
            "start_date",
            "due_date",
            "completion_date",
            "progress",
            "status",
            "priority",
        ],
        "metrics": [],
        "planned_metrics": [
            "project_count",
            "completion_rate",
            "delay_days",
            "average_progress",
        ],
    },

    # -----------------------------------------------------
    # INVENTORY
    # -----------------------------------------------------

    "inventory": {
        "label": "انبار و موجودی",
        "aliases": [
            "inventory",
            "warehouse",
            "stock",
            "انبار",
            "موجودی",
            "مدیریت موجودی",
        ],
        "column_hints": [
            "stock",
            "stock_quantity",
            "inventory_id",
            "warehouse_id",
            "sku",
            "reorder_level",
            "supplier",
            "product_id",
        ],
        "metrics": [],
        "planned_metrics": [
            "stock_quantity",
            "low_stock_count",
            "inventory_value",
            "stock_turnover",
        ],
    },

    # -----------------------------------------------------
    # MARKETING
    # -----------------------------------------------------

    "marketing": {
        "label": "بازاریابی",
        "aliases": [
            "marketing",
            "campaign",
            "advertising",
            "بازاریابی",
            "تبلیغات",
            "کمپین",
        ],
        "column_hints": [
            "campaign_id",
            "campaign_name",
            "impressions",
            "clicks",
            "conversions",
            "leads",
            "ad_spend",
            "channel",
        ],
        "metrics": [],
        "planned_metrics": [
            "conversion_rate",
            "click_through_rate",
            "cost_per_lead",
            "campaign_roi",
        ],
    },

    # -----------------------------------------------------
    # GENERIC
    # -----------------------------------------------------

    "generic": {
        "label": "عمومی",
        "aliases": [
            "general",
            "generic",
            "عمومی",
            "سایر",
        ],
        "column_hints": [],
        "metrics": [],
        "planned_metrics": [],
    },
}


# =========================================================
# 2. TEXT NORMALIZATION
# =========================================================

def normalize_domain_text(value: Any) -> str:
    """
    Normalize Persian and English text.

    Handles:
    - Unicode normalization
    - Persian and Arabic character variants
    - Persian and Arabic digits
    - Zero-width characters
    - Whitespace
    - English case
    """

    if value is None:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value),
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
            str(index),
        )

        value = value.replace(
            arabic_digits[index],
            str(index),
        )

    value = value.lower().strip()

    value = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        value,
    )

    return " ".join(value.split())


def normalize_domain_column(value: Any) -> str:
    """
    Normalize a column name for comparison.

    Examples:
        employee id -> employee_id
        Employee-ID -> employee_id
        EMPLOYEE_ID -> employee_id

    Does not rename actual DataFrame columns.
    """

    value = normalize_domain_text(value)

    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


# =========================================================
# 3. DOMAIN ACCESS
# =========================================================

def get_domain_definition(
    domain: str,
) -> Optional[Dict[str, Any]]:
    """Return a domain definition."""

    if not domain:
        return None

    return DOMAIN_REGISTRY.get(domain)


def get_supported_domains() -> List[str]:
    """Return all registered domain names."""

    return list(DOMAIN_REGISTRY.keys())


def get_domain_metrics(
    domain: str,
) -> List[str]:
    """
    Return implemented metric names for a domain.

    Planned but unimplemented metrics are excluded.
    """

    definition = get_domain_definition(domain)

    if not definition:
        return []

    return list(
        definition.get("metrics", [])
    )


def get_planned_domain_metrics(
    domain: str,
) -> List[str]:
    """
    Return planned metric names.

    These metrics are not necessarily executable yet.
    """

    definition = get_domain_definition(domain)

    if not definition:
        return []

    return list(
        definition.get("planned_metrics", [])
    )


def get_domain_aliases(
    domain: str,
) -> List[str]:

    definition = get_domain_definition(domain)

    if not definition:
        return []

    return list(
        definition.get("aliases", [])
    )


# =========================================================
# 4. DOMAIN DETECTION
# =========================================================

def detect_domain(
    columns,
    schema=None,
) -> Dict[str, Any]:
    """
    Detect the most likely business domain.

    Parameters:
        columns:
            DataFrame column names.

        schema:
            Reserved for future schema-aware detection.
            Current detection uses column names.

    Returns:
        {
            domain,
            label,
            confidence,
            matched_columns,
            candidate_domains
        }

    Note:
        Confidence is a relative heuristic score.
        It is not a statistical probability.
    """

    if columns is None:
        columns = []

    normalized_columns = {}

    for column in columns:

        normalized = normalize_domain_column(column)

        if normalized:
            normalized_columns[normalized] = str(column)

    results = []

    for domain_name, definition in DOMAIN_REGISTRY.items():

        if domain_name == "generic":
            continue

        matched_columns = []
        matched_hints = []

        hints = definition.get(
            "column_hints",
            [],
        )

        normalized_hints = {
            normalize_domain_column(hint)
            for hint in hints
        }

        for hint in normalized_hints:

            if hint in normalized_columns:

                original_column = normalized_columns[hint]

                matched_columns.append(original_column)
                matched_hints.append(hint)

        if not matched_columns:
            continue

        score = len(matched_columns)

        results.append({
            "domain": domain_name,
            "label": definition["label"],
            "score": score,
            "matched_columns": matched_columns,
            "matched_hints": matched_hints,
            "hint_count": len(normalized_hints),
        })

    results.sort(
        key=lambda item: (
            item["score"],
            item["domain"],
        ),
        reverse=True,
    )

    if not results:

        return {
            "domain": "generic",
            "label": DOMAIN_REGISTRY["generic"]["label"],
            "confidence": 0.0,
            "matched_columns": [],
            "candidate_domains": [],
        }

    best = results[0]

    total_matches = sum(
        item["score"]
        for item in results
    )

    confidence = round(
        best["score"] / total_matches,
        2,
    )

    return {
        "domain": best["domain"],
        "label": best["label"],
        "confidence": confidence,
        "matched_columns": best["matched_columns"],
        "candidate_domains": results,
    }


# =========================================================
# 5. DOMAIN VALIDATION
# =========================================================

def validate_domain_registry() -> Dict[str, Any]:
    """
    Validate domain definitions and metric references.
    """

    errors = []
    warnings = []

    for domain_name, definition in DOMAIN_REGISTRY.items():

        if not definition.get("label"):
            errors.append(
                f"{domain_name}: missing label"
            )

        if not isinstance(
            definition.get("aliases", []),
            list,
        ):
            errors.append(
                f"{domain_name}: aliases must be a list"
            )

        if not isinstance(
            definition.get("column_hints", []),
            list,
        ):
            errors.append(
                f"{domain_name}: column_hints must be a list"
            )

        if not isinstance(
            definition.get("metrics", []),
            list,
        ):
            errors.append(
                f"{domain_name}: metrics must be a list"
            )

        if not isinstance(
            definition.get("planned_metrics", []),
            list,
        ):
            errors.append(
                f"{domain_name}: planned_metrics must be a list"
            )

        if (
            not definition.get("column_hints")
            and domain_name != "generic"
        ):
            warnings.append(
                f"{domain_name}: no column hints defined"
            )

        implemented = definition.get("metrics", [])
        planned = definition.get("planned_metrics", [])

        overlap = set(implemented) & set(planned)

        if overlap:
            errors.append(
                f"{domain_name}: metrics cannot be both "
                f"implemented and planned: {sorted(overlap)}"
            )

    # Check consistency with Metric Registry.
    try:

        from src.metric_registry import (
            DOMAIN_METRIC_REGISTRY,
        )

        for domain_name, definition in DOMAIN_REGISTRY.items():

            for metric_name in definition.get("metrics", []):

                metric_definition = DOMAIN_METRIC_REGISTRY.get(
                    metric_name
                )

                if metric_definition is None:
                    errors.append(
                        f"{domain_name}: metric "
                        f"'{metric_name}' is not defined "
                        f"in metric_registry.py"
                    )
                    continue

                if metric_definition.get("domain") != domain_name:
                    errors.append(
                        f"{metric_name}: domain mismatch "
                        f"between registries"
                    )

        for metric_name, definition in DOMAIN_METRIC_REGISTRY.items():

            domain_name = definition.get("domain")

            if domain_name not in DOMAIN_REGISTRY:
                errors.append(
                    f"{metric_name}: unknown domain "
                    f"'{domain_name}'"
                )
                continue

            registered_metrics = DOMAIN_REGISTRY[
                domain_name
            ].get("metrics", [])

            if metric_name not in registered_metrics:
                errors.append(
                    f"{metric_name}: missing from "
                    f"DOMAIN_REGISTRY['{domain_name}']['metrics']"
                )

    except ImportError as exc:

        warnings.append(
            f"Could not cross-check Metric Registry: {exc}"
        )

    return {
        "valid": len(errors) == 0,
        "total_domains": len(DOMAIN_REGISTRY),
        "domains": get_supported_domains(),
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# 6. PRINT DOMAIN RESULT
# =========================================================

def print_domain_result(
    result: Dict[str, Any],
) -> None:

    print("\n========== DOMAIN DETECTION ==========")

    print(
        f"Domain: {result.get('domain')}"
    )

    print(
        f"Label: {result.get('label')}"
    )

    print(
        f"Confidence: {result.get('confidence')}"
    )

    print(
        f"Matched Columns: {result.get('matched_columns', [])}"
    )

    print("\nCandidate Domains:")

    candidates = result.get(
        "candidate_domains",
        [],
    )

    if not candidates:
        print("- No specific domain detected.")

    for candidate in candidates:

        print(
            f"- {candidate['domain']}: "
            f"{candidate['score']} matches"
        )

        print(
            f"  Columns: {candidate['matched_columns']}"
        )

    print("=======================================")


# =========================================================
# 7. PRINT REGISTRY SUMMARY
# =========================================================

def print_domain_registry_summary() -> None:

    validation = validate_domain_registry()

    print("\n========== DOMAIN REGISTRY ==========")

    print(
        f"Total Domains: {validation['total_domains']}"
    )

    print(
        f"Domains: {', '.join(validation['domains'])}"
    )

    print(
        f"Valid: {validation['valid']}"
    )

    if validation["errors"]:

        print("\nErrors:")

        for error in validation["errors"]:
            print(f"- {error}")

    if validation["warnings"]:

        print("\nWarnings:")

        for warning in validation["warnings"]:
            print(f"- {warning}")

    print("=====================================")
