
import pandas as pd
import numpy as np


def analyze_data(df: pd.DataFrame) -> dict:
    """
    Generate a statistical profile of a dataset.
    """

    rows, columns = df.shape

    profile = {
        "dataset_info": {
            "rows": int(rows),
            "columns": int(columns),
            "duplicate_rows": int(df.duplicated().sum())
        },
        "column_analysis": {}
    }

    for col in df.columns:

        series = df[col]

        missing = int(series.isna().sum())
        non_null = series.dropna()

        info = {
            "data_type": str(series.dtype),
            "missing_count": missing,
            "missing_percentage": round(
                missing / rows * 100, 2
            ),
            "unique_values": int(series.nunique()),
        }

        # Numeric analysis
        if pd.api.types.is_numeric_dtype(series):

            info["type"] = "numeric"

            info["statistics"] = {
                "mean": _safe_float(series.mean()),
                "median": _safe_float(series.median()),
                "std": _safe_float(series.std()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "q1": _safe_float(series.quantile(0.25)),
                "q3": _safe_float(series.quantile(0.75))
            }

            # IQR outlier detection
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if pd.notna(iqr) and iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outliers = (
                    (series < lower) |
                    (series > upper)
                ).sum()

                info["outliers"] = int(outliers)
            else:
                info["outliers"] = 0

        # Datetime detection
        elif (
            pd.api.types.is_datetime64_any_dtype(series)
            or (
                series.dtype == "object"
                and "date" in str(col).lower()
            )
        ):

            parsed = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid = parsed.dropna()

            if len(valid) >= max(1, int(len(non_null) * 0.8)):

                info["type"] = "datetime"

                info["date_range"] = {
                    "min": str(valid.min()),
                    "max": str(valid.max())
                }

            else:
                info["type"] = "categorical"

        else:
            info["type"] = "categorical"

        # Categorical analysis
        if info["type"] == "categorical":

            top = series.value_counts(dropna=True).head(5)

            info["top_values"] = [
                {
                    "value": str(value),
                    "frequency": int(count)
                }
                for value, count in top.items()
            ]

        profile["column_analysis"][str(col)] = info

    return profile


import math

def _safe_float(value):
    if pd.isna(value) or not math.isfinite(float(value)):
        return None

    return float(value)

