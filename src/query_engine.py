
"""
General-Purpose Query Engine

Responsibilities:
- Execute generic analytical operations.
- Calculate registry-defined metrics.
- Support computed metrics.
- Aggregate and group data.
- Apply safe filters.
- Analyze time-based trends.

All numerical calculations are performed by Pandas.
"""

import pandas as pd

from src.metric_registry import (
    METRIC_REGISTRY,
    get_metric,
)


# =====================================================
# CONSTANTS
# =====================================================

SUPPORTED_OPERATIONS = {
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

NUMERIC_OPERATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
}

GROUP_AGGREGATIONS = {
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "nunique",
}

FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
}

BOOLEAN_TRUE = {
    "yes", "true", "1", "y"
}

BOOLEAN_FALSE = {
    "no", "false", "0", "n"
}

TREND_GRANULARITIES = {
    "day",
    "week",
    "month",
    "quarter",
    "year",
}


# =====================================================
# GENERAL VALIDATION
# =====================================================

def validate_dataframe(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")


def validate_column(df, column):

    if not isinstance(column, str):
        raise ValueError("A valid column name is required.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in dataset."
        )


def validate_group_column(df, group_by):

    if group_by is not None:
        validate_column(df, group_by)


def validate_sort(sort):

    if sort not in ("asc", "desc"):
        raise ValueError("Sort must be 'asc' or 'desc'.")


def validate_top_k(top_k):

    if top_k is None:
        return

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")


# =====================================================
# JSON SAFE CONVERSION
# =====================================================

def safe_value(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def safe_records(df):

    records = df.to_dict(orient="records")

    return [
        {
            key: safe_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


# =====================================================
# METRIC ENGINE
# =====================================================

def get_metric_series(df, metric_name):

    definition = get_metric(metric_name)

    if definition is None:
        raise ValueError(
            f"Unknown metric: {metric_name}"
        )

    required_columns = definition.get(
        "required_columns", []
    )

    optional_columns = definition.get(
        "optional_columns", []
    )

    for column in required_columns:
        validate_column(df, column)

    calculation = definition.get("calculation")

    # -------------------------------------------------
    # Computed metric: Sales revenue
    # -------------------------------------------------

    if calculation == "sales_revenue":

        quantity = pd.to_numeric(
            df[required_columns[0]],
            errors="coerce"
        )

        price = pd.to_numeric(
            df[required_columns[1]],
            errors="coerce"
        )

        if optional_columns and optional_columns[0] in df.columns:

            discount = pd.to_numeric(
                df[optional_columns[0]],
                errors="coerce"
            ).fillna(0)

        else:
            discount = pd.Series(
                0,
                index=df.index,
                dtype="float64"
            )

        valid = (
            quantity.notna()
            & price.notna()
            & discount.notna()
            & (quantity >= 0)
            & (price >= 0)
            & discount.between(0, 100)
        )

        result = pd.Series(
            float("nan"),
            index=df.index,
            dtype="float64"
        )

        result.loc[valid] = (
            quantity.loc[valid]
            * price.loc[valid]
            * (1 - discount.loc[valid] / 100)
        )

        return result

    # -------------------------------------------------
    # Sum of a column
    # -------------------------------------------------

    if calculation == "column_sum":

        column = required_columns[0]

        return pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -------------------------------------------------
    # Mean of a column
    # -------------------------------------------------

    if calculation == "column_mean":

        column = required_columns[0]

        return pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -------------------------------------------------
    # Unique count metric
    # -------------------------------------------------

    if calculation == "unique_count":

        column = required_columns[0]

        return df[column]

    # -------------------------------------------------
    # Generic row count
    # -------------------------------------------------

    if calculation == "row_count":

        return pd.Series(
            1,
            index=df.index,
            dtype="int64"
        )

    raise ValueError(
        f"Unsupported metric calculation: {calculation}"
    )


def get_boolean_status(series):

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="string"
    )

    result.loc[
        normalized.isin(BOOLEAN_TRUE)
    ] = "Yes"

    result.loc[
        normalized.isin(BOOLEAN_FALSE)
    ] = "No"

    return result


def calculate_boolean_percentage(
    df,
    column,
    group_by=None,
    sort="desc",
    top_k=None
):

    validate_column(df, column)
    validate_group_column(df, group_by)
    validate_sort(sort)
    validate_top_k(top_k)

    working = pd.DataFrame(index=df.index)

    working["_status"] = get_boolean_status(
        df[column]
    )

    if group_by is not None:
        working["_group"] = df[group_by]

    # Unknown values are excluded from the denominator.
    valid = working.dropna(subset=["_status"])

    if valid.empty:
        raise ValueError(
            "No valid Boolean values were found."
        )

    if group_by is None:

        total = len(valid)

        positive = int(
            (valid["_status"] == "Yes").sum()
        )

        return {
            "operation": "return_rate",
            "column": column,
            "total_valid_records": int(total),
            "positive_records": positive,
            "percentage": round(
                positive / total * 100, 2
            ),
            "return_rate_pct": round(
                positive / total * 100, 2
            ),
        }

    valid = valid.dropna(subset=["_group"])

    if valid.empty:
        raise ValueError(
            "No valid grouping records were found."
        )

    grouped = (
        valid.groupby("_group", dropna=True)
        .agg(
            total_records=("_status", "count"),
            positive_records=(
                "_status",
                lambda values: int(
                    (values == "Yes").sum()
                )
            ),
        )
        .reset_index()
    )

    grouped["percentage"] = (
        grouped["positive_records"]
        / grouped["total_records"]
        * 100
    )

    grouped["return_rate_pct"] = grouped["percentage"]

    grouped = grouped.sort_values(
        "percentage",
        ascending=(sort == "asc")
    )

    total_groups = len(grouped)

    if top_k is not None:
        grouped = grouped.head(top_k)

    records = []

    for record in grouped.to_dict(orient="records"):

        records.append({
            group_by: safe_value(record["_group"]),
            "total_records": int(record["total_records"]),
            "positive_records": int(record["positive_records"]),
            "percentage": round(
                float(record["percentage"]), 2
            ),
            "return_rate_pct": round(
                float(record["return_rate_pct"]), 2
            ),
        })

    return {
        "operation": "return_rate",
        "column": column,
        "group_by": group_by,
        "total_groups": int(total_groups),
        "total_valid_records": int(len(valid)),
        "results": records,
    }


# =====================================================
# METRIC RESOLUTION
# =====================================================

def resolve_query_metric(query):

    metric_name = query.get("metric")

    if metric_name in METRIC_REGISTRY:
        return metric_name

    column = query.get("column")

    if column in METRIC_REGISTRY:

        definition = get_metric(column)

        if definition.get("calculation") not in {
            "sum",
            "mean",
            "median",
            "min",
            "max",
        }:
            return column

    return None


def get_query_series(df, query):

    metric_name = resolve_query_metric(query)

    if metric_name is not None:

        definition = get_metric(metric_name)

        if definition.get("calculation") == "boolean_percentage":
            raise ValueError(
                "Boolean metrics must use the Boolean metric engine."
            )

        return (
            get_metric_series(df, metric_name),
            metric_name,
        )

    column = query.get("column")

    validate_column(df, column)

    series = df[column]

    return series, column


# =====================================================
# AGGREGATION ENGINE
# =====================================================

def aggregate_series(series, aggregation):

    if aggregation not in GROUP_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    if aggregation == "count":
        return int(series.count())

    if aggregation == "nunique":
        return int(series.nunique())

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if numeric.empty:
        return None

    calculations = {
        "sum": numeric.sum,
        "mean": numeric.mean,
        "median": numeric.median,
        "min": numeric.min,
        "max": numeric.max,
    }

    result = calculations[aggregation]()

    return safe_value(result)


def calculate_statistic(df, query, operation):

    series, metric_name = get_query_series(
        df,
        query
    )

    if operation == "count":

        result = int(series.count())

    else:

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if numeric.empty:
            raise ValueError(
                "No valid numeric values are available."
            )

        calculations = {
            "sum": numeric.sum,
            "mean": numeric.mean,
            "median": numeric.median,
            "min": numeric.min,
            "max": numeric.max,
        }

        result = safe_value(
            calculations[operation]()
        )

    return {
        "operation": operation,
        "metric": metric_name,
        "column": query.get("column"),
        "valid_values": int(series.count()),
        "result": result,
    }


# =====================================================
# GROUP BY ENGINE
# =====================================================

def calculate_groupby(df, query):

    group_by = query.get("group_by")

    validate_column(df, group_by)

    aggregation = query.get("aggregation") or "sum"

    sort = query.get("sort") or "desc"

    top_k = query.get("top_k")

    validate_sort(sort)
    validate_top_k(top_k)

    metric_name = resolve_query_metric(query)

    # Boolean metric
    if metric_name is not None:

        definition = get_metric(metric_name)

        if definition.get("calculation") == "boolean_percentage":

            column = definition["required_columns"][0]

            return calculate_boolean_percentage(
                df=df,
                column=column,
                group_by=group_by,
                sort=sort,
                top_k=top_k,
            )

    series, metric_name = get_query_series(
        df,
        query
    )

    working = pd.DataFrame(index=df.index)

    working["_group"] = df[group_by]
    working["_value"] = series

    working = working.dropna(subset=["_group"])

    if aggregation not in GROUP_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    grouped = working.groupby(
        "_group",
        dropna=True
    )["_value"]

    if aggregation in ("count", "nunique"):

        result = grouped.agg(aggregation)

    else:

        numeric = pd.to_numeric(
            working["_value"],
            errors="coerce"
        )

        working["_value"] = numeric

        grouped = working.groupby(
            "_group",
            dropna=True
        )["_value"]

        result = grouped.agg(aggregation)

    result = result.reset_index(name="result")

    result = result.sort_values(
        "result",
        ascending=(sort == "asc"),
        na_position="last"
    )

    total_groups = len(result)

    if top_k is not None:
        result = result.head(top_k)

    records = []

    for record in result.to_dict(orient="records"):

        records.append({
            group_by: safe_value(record["_group"]),
            "result": safe_value(record["result"]),
        })

    return {
        "operation": "groupby",
        "metric": metric_name,
        "column": query.get("column"),
        "group_by": group_by,
        "aggregation": aggregation,
        "total_groups": int(total_groups),
        "results": records,
    }


# =====================================================
# COUNT ENGINE
# =====================================================
def calculate_count(df, query):

    metric_name = resolve_query_metric(query)

    # Registered analytical metric
    if metric_name:

        definition = get_metric(metric_name)

        if definition is None:
            raise ValueError(
                f"Unknown metric: {metric_name}"
            )

        series, _ = get_query_series(
            df,
            query
        )

        if definition.get("calculation") == "unique_count":
            count = int(series.nunique())

        else:
            count = int(series.count())

        return {
            "operation": "count",
            "metric": metric_name,
            "count": count,
        }

    # Generic row count
    column = query.get("column")

    if column is None:
        return {
            "operation": "count",
            "description": "Total number of rows",
            "count": int(len(df)),
        }

    validate_column(df, column)

    series = df[column]
    aggregation = query.get("aggregation")

    # Count distinct values
    if aggregation == "nunique":
        count = int(series.nunique())

    # Count non-null values
    else:
        count = int(series.count())

    return {
        "operation": "count",
        "column": column,
        "aggregation": aggregation,
        "count": count,
    }

# =====================================================
# FILTER ENGINE
# =====================================================

def calculate_filter(df, query):

    condition = query.get("filter")

    if not isinstance(condition, dict):
        raise ValueError(
            "A filter condition is required."
        )

    column = condition.get("column")
    operator = condition.get("operator")
    value = condition.get("value")

    validate_column(df, column)

    if operator not in FILTER_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator: {operator}"
        )

    series = df[column]

    if operator == "contains":

        if not isinstance(value, str):
            raise ValueError(
                "Contains requires a text value."
            )

        mask = series.astype("string").str.contains(
            value,
            case=False,
            na=False,
            regex=False,
        )

    elif operator in ("eq", "ne"):

        if pd.api.types.is_numeric_dtype(series):

            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Numeric filter value required."
                )

        if operator == "eq":
            mask = series.eq(value)
        else:
            mask = series.ne(value)

    else:

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                "Comparison operators require a numeric column."
            )

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Numeric filter value required."
            )

        comparisons = {
            "gt": series.gt,
            "gte": series.ge,
            "lt": series.lt,
            "lte": series.le,
        }

        mask = comparisons[operator](value)

    filtered = df.loc[mask.fillna(False)]

    top_k = query.get("top_k")

    if top_k is None:
        top_k = 10

    validate_top_k(top_k)

    return {
        "operation": "filter",
        "filter": condition,
        "total_matches": int(len(filtered)),
        "displayed_records": min(
            len(filtered), top_k
        ),
        "results": safe_records(
            filtered.head(top_k)
        ),
    }


# =====================================================
# RETURN RATE COMPATIBILITY
# =====================================================

def calculate_return_rate(df, query):

    metric_name = resolve_query_metric(query)

    if metric_name is not None:

        definition = get_metric(metric_name)

        if definition.get("calculation") == "boolean_percentage":

            column = definition["required_columns"][0]

            return calculate_boolean_percentage(
                df=df,
                column=column,
                group_by=query.get("group_by"),
                sort=query.get("sort") or "desc",
                top_k=query.get("top_k"),
            )

    column = query.get("column")

    if column is None:
        raise ValueError(
            "A Boolean status column is required."
        )

    return calculate_boolean_percentage(
        df=df,
        column=column,
        group_by=query.get("group_by"),
        sort=query.get("sort") or "desc",
        top_k=query.get("top_k"),
    )


# =====================================================
# TREND ENGINE
# =====================================================

def calculate_trend_by_date(df, query):
    """
    General-purpose time trend analysis.

    Preserves the legacy output keys:
    metrics, results, metric_analysis, rankings.
    """

    date_column = query.get("date_column")

    validate_column(df, date_column)

    granularity = query.get("granularity") or "month"

    if granularity not in TREND_GRANULARITIES:
        raise ValueError(
            "Unsupported time granularity."
        )

    ranking = query.get("ranking")

    if ranking not in (None, "asc", "desc"):
        raise ValueError(
            "Ranking must be asc, desc or None."
        )

    requested_metrics = query.get("metrics")

    if not requested_metrics:

        requested_metrics = [
            name
            for name, definition in METRIC_REGISTRY.items()
            if (
                definition.get("domain") == "generic"
                and definition.get("calculation")
                in {
                    "column_sum",
                    "column_mean",
                    "sales_revenue",
                }
            )
        ]

    metrics = []

    for name in requested_metrics:

        if name not in METRIC_REGISTRY:
            continue

        definition = get_metric(name)

        if definition.get("calculation") == "boolean_percentage":
            continue

        try:
            get_metric_series(df, name)
            metrics.append(name)
        except (ValueError, KeyError):
            continue

    if not metrics:

        # Backward compatibility with the sales dataset.
        metrics = [
            name
            for name in ("quantity", "revenue", "orders")
            if name in METRIC_REGISTRY
        ]

    working = df.copy()

    working["_date"] = pd.to_datetime(
        working[date_column],
        errors="coerce"
    )

    working = working.dropna(
        subset=["_date"]
    )

    if working.empty:
        raise ValueError(
            "No valid dates available."
        )

    if granularity == "day":

        working["_period"] = (
            working["_date"].dt.strftime("%Y-%m-%d")
        )

    elif granularity == "week":

        working["_period"] = (
            working["_date"].dt.to_period("W").astype(str)
        )

    elif granularity == "month":

        working["_period"] = (
            working["_date"].dt.to_period("M").astype(str)
        )

    elif granularity == "quarter":

        working["_period"] = (
            working["_date"].dt.to_period("Q").astype(str)
        )

    else:

        working["_period"] = (
            working["_date"].dt.strftime("%Y")
        )

    metric_results = {}

    for metric in metrics:

        definition = get_metric(metric)
        calculation = definition.get("calculation")

        if calculation == "unique_count":

            column = definition["required_columns"][0]

            validate_column(working, column)

            values = working[column]

            aggregated = (
                pd.DataFrame({
                    "_period": working["_period"],
                    "_value": values,
                })
                .groupby("_period")["_value"]
                .nunique()
            )

        else:

            values = get_metric_series(
                working,
                metric
            )

            temp = pd.DataFrame({
                "_period": working["_period"],
                "_value": values,
            })

            temp["_value"] = pd.to_numeric(
                temp["_value"],
                errors="coerce"
            )

            aggregated = (
                temp.groupby("_period")["_value"]
                .sum(min_count=1)
            )

        metric_results[metric] = aggregated

    result = pd.DataFrame(metric_results)

    result = result.sort_index()

    if result.empty:
        raise ValueError(
            "No analytical trend results available."
        )

    # Keep periods with missing metric values as null.
    records = []

    for period, row in result.iterrows():

        record = {
            "period": str(period)
        }

        for metric in metrics:

            value = row[metric]

            if pd.isna(value):
                record[metric] = None

            elif metric == "orders":
                record[metric] = int(value)

            else:
                record[metric] = round(
                    float(value), 2
                )

        records.append(record)

    # -------------------------------------------------
    # Trend analysis
    # -------------------------------------------------

    metric_analysis = {}

    for metric in metrics:

        series = result[metric].dropna()

        if series.empty:
            continue

        highest_value = series.max()
        lowest_value = series.min()

        highest_periods = [
            str(period)
            for period in series.index[
                series == highest_value
            ]
        ]

        lowest_periods = [
            str(period)
            for period in series.index[
                series == lowest_value
            ]
        ]

        period_changes = []

        for index in range(1, len(series)):

            previous_period = series.index[index - 1]
            current_period = series.index[index]

            previous_value = series.iloc[index - 1]
            current_value = series.iloc[index]

            difference = (
                current_value - previous_value
            )

            if difference > 0:
                direction = "increase"
            elif difference < 0:
                direction = "decrease"
            else:
                direction = "unchanged"

            percentage_change = None

            if previous_value != 0:

                percentage_change = round(
                    float(
                        difference
                        / abs(previous_value)
                        * 100
                    ),
                    2,
                )

            period_changes.append({
                "from_period": str(previous_period),
                "to_period": str(current_period),
                "previous_value": safe_value(previous_value),
                "current_value": safe_value(current_value),
                "absolute_change": safe_value(difference),
                "percentage_change": percentage_change,
                "percentage_change_status": (
                    "calculated"
                    if percentage_change is not None
                    else "undefined_due_to_zero_previous_value"
                ),
                "direction": direction,
            })

        metric_analysis[metric] = {
            "highest": {
                "periods": highest_periods,
                "value": safe_value(highest_value),
            },
            "lowest": {
                "periods": lowest_periods,
                "value": safe_value(lowest_value),
            },
            "period_changes": period_changes,
        }

    # -------------------------------------------------
    # Optional rankings
    # -------------------------------------------------

    rankings = {}

    if ranking is not None:

        ascending = ranking == "asc"

        for metric in metrics:

            ranked = result[metric].dropna().sort_values(
                ascending=ascending
            )

            rankings[metric] = [
                {
                    "period": str(period),
                    "value": safe_value(value),
                }
                for period, value in ranked.items()
            ]

    return {
        "operation": "trend_by_date",
        "date_column": date_column,
        "granularity": granularity,
        "total_periods": int(len(result)),
        "valid_date_records": int(len(working)),
        "metrics": metrics,
        "results": records,
        "metric_analysis": metric_analysis,
        "rankings": rankings,
    }


# =====================================================
# MAIN QUERY EXECUTOR
# =====================================================

def execute_query(df: pd.DataFrame, query: dict) -> dict:

    validate_dataframe(df)

    if not isinstance(query, dict):
        raise ValueError(
            "Query must be a dictionary."
        )

    operation = query.get("operation")

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # Trend
    if operation == "trend_by_date":

        return calculate_trend_by_date(
            df,
            query
        )

    # Filter
    if operation == "filter":

        return calculate_filter(
            df,
            query
        )

    # Boolean metric
    metric_name = resolve_query_metric(query)

    if metric_name is not None:

        definition = get_metric(metric_name)

        if definition.get("calculation") == "boolean_percentage":

            return calculate_boolean_percentage(
                df=df,
                column=definition["required_columns"][0],
                group_by=query.get("group_by"),
                sort=query.get("sort") or "desc",
                top_k=query.get("top_k"),
            )

    # Explicit return-rate operation
    if operation == "return_rate":

        return calculate_return_rate(
            df,
            query
        )

    # Grouping takes precedence over standalone aggregation.
    if operation == "groupby" or query.get("group_by") is not None:

        return calculate_groupby(
            df,
            query
        )

    # Count
    if operation == "count":

        return calculate_count(
            df,
            query
        )

    # Numeric operations
    if operation in NUMERIC_OPERATIONS:

        return calculate_statistic(
            df,
            query,
            operation
        )

    raise ValueError(
        f"Operation not implemented: {operation}"
    )
