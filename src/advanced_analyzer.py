
import pandas as pd
import numpy as np


def analyze_advanced(df):

    data = df.copy()
    result = {}

    # --------------------------------
    # 1. Date Analysis
    # --------------------------------

    if "order_date" in data.columns:

        dates = pd.to_datetime(
            data["order_date"],
            errors="coerce"
        )

        valid_dates = dates.dropna()

        result["date_analysis"] = {
            "valid_dates": int(len(valid_dates)),
            "invalid_dates": int(dates.isna().sum()),

            "start_date": (
                valid_dates.min().strftime("%Y-%m-%d")
                if len(valid_dates) else None
            ),

            "end_date": (
                valid_dates.max().strftime("%Y-%m-%d")
                if len(valid_dates) else None
            )
        }

        if len(valid_dates):

            monthly_counts = (
                valid_dates.dt.to_period("M")
                .value_counts()
                .sort_index()
            )

            result["date_analysis"]["orders_by_month"] = {
                str(month): int(count)
                for month, count in monthly_counts.items()
            }

    # --------------------------------
    # 2. Data Quality Validation
    # --------------------------------

    quality = {}

    if "quantity" in data.columns:

        invalid = (
            data["quantity"].notna()
            & (data["quantity"] <= 0)
        )

        quality["invalid_quantity_rows"] = int(
            invalid.sum()
        )

    if "unit_price_usd" in data.columns:

        invalid = (
            data["unit_price_usd"].notna()
            & (data["unit_price_usd"] < 0)
        )

        quality["negative_price_rows"] = int(
            invalid.sum()
        )

    if "discount_pct" in data.columns:

        invalid = (
            data["discount_pct"].notna()
            & ~data["discount_pct"].between(0, 100)
        )

        quality["invalid_discount_rows"] = int(
            invalid.sum()
        )

    if "customer_rating" in data.columns:

        invalid = (
            data["customer_rating"].notna()
            & ~data["customer_rating"].between(1, 5)
        )

        quality["invalid_rating_rows"] = int(
            invalid.sum()
        )

    result["data_quality"] = quality

    # --------------------------------
    # 3. Business Metrics
    # --------------------------------

    required_columns = {
        "quantity",
        "unit_price_usd",
        "discount_pct"
    }

    if required_columns.issubset(data.columns):

        valid = (
            data["quantity"].notna()
            & data["unit_price_usd"].notna()
            & data["discount_pct"].notna()
            & (data["quantity"] > 0)
            & (data["unit_price_usd"] >= 0)
            & data["discount_pct"].between(0, 100)
        )

        valid_orders = data.loc[valid].copy()

        valid_orders["estimated_order_value"] = (
            valid_orders["quantity"]
            * valid_orders["unit_price_usd"]
            * (
                1 - valid_orders["discount_pct"] / 100
            )
        )

        result["business_metrics"] = {

            "valid_orders": int(len(valid_orders)),

            "excluded_orders": int((~valid).sum()),

            "total_estimated_order_value": round(
                float(
                    valid_orders["estimated_order_value"].sum()
                ), 2
            ),

            "average_estimated_order_value": (
                round(
                    float(
                        valid_orders["estimated_order_value"].mean()
                    ), 2
                )
                if len(valid_orders) else None
            )
        }

    # --------------------------------
    # 4. Return Analysis
    # --------------------------------

    if "returned" in data.columns:

        returned = (
            data["returned"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        yes = returned.isin([
            "yes",
            "true",
            "1"
        ])

        no = returned.isin([
            "no",
            "false",
            "0"
        ])

        known = yes | no

        total = int(known.sum())

        result["return_analysis"] = {

            "returned_orders": int(yes.sum()),

            "non_returned_orders": int(no.sum()),

            "unknown_status": int((~known).sum()),

            "return_rate_pct": (
                round(
                    float(yes.sum() / total * 100), 2
                )
                if total else None
            )
        }

        # Only valid return statuses are used
        known_data = data.loc[known].copy()

        known_data["_returned"] = (
            yes.loc[known].astype(int)
        )

        # Returns by city
        if "city" in data.columns:

            city_returns = (
                known_data.groupby("city")["_returned"]
                .agg(
                    returned_orders="sum",
                    known_orders="count"
                )
            )

            city_returns["return_rate_pct"] = (
                city_returns["returned_orders"]
                / city_returns["known_orders"] * 100
            ).round(2)

            result["returns_by_city"] = (
                city_returns.reset_index()
                .to_dict(orient="records")
            )

        # Returns by category
        if "category" in data.columns:

            category_returns = (
                known_data.groupby("category")["_returned"]
                .agg(
                    returned_orders="sum",
                    known_orders="count"
                )
            )

            category_returns["return_rate_pct"] = (
                category_returns["returned_orders"]
                / category_returns["known_orders"] * 100
            ).round(2)

            result["returns_by_category"] = (
                category_returns.reset_index()
                .to_dict(orient="records")
            )

    # --------------------------------
    # 5. Correlation Analysis
    # --------------------------------

    # Exclude identifiers, but include customer_rating
    excluded_columns = {
        "order_id"
    }

    numeric_columns = [
        col
        for col in data.select_dtypes(
            include="number"
        ).columns
        if col not in excluded_columns
    ]

    if len(numeric_columns) >= 2:

        corr = data[numeric_columns].corr(
            method="spearman"
        )

        result["correlations"] = {

            str(column): {

                str(other_column): (
                    None
                    if pd.isna(value)
                    else round(float(value), 3)
                )

                for other_column, value in corr[column].items()
            }

            for column in corr.columns
        }

    # --------------------------------
    # 6. Return Result
    # --------------------------------

    return result
