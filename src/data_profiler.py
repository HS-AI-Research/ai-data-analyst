
import pandas as pd
import numpy as np


def profile_dataset(df, sample_size=5):
    """
    Analyze the structure and quality of any Pandas DataFrame.

    Returns:
        dict: Dataset profile information.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    total_rows = len(df)
    total_columns = len(df.columns)

    columns_profile = []

    for column in df.columns:

        series = df[column]

        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        sample_values = (
            series.dropna()
            .head(sample_size)
            .tolist()
        )

        if pd.api.types.is_datetime64_any_dtype(series):
            data_type = "datetime"

        elif pd.api.types.is_bool_dtype(series):
            data_type = "boolean"

        elif pd.api.types.is_numeric_dtype(series):
            data_type = "numeric"

        else:
            data_type = "text"

        columns_profile.append({
            "column": str(column),
            "data_type": data_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / total_rows * 100, 2
            ),
            "unique_count": unique_count,
            "unique_percentage": round(
                unique_count / total_rows * 100, 2
            ),
            "sample_values": [
                str(value) if isinstance(value, (np.integer, np.floating))
                else str(value)
                for value in sample_values
            ]
        })

    return {
        "dataset": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_cells": total_rows * total_columns,
            "duplicate_rows": int(df.duplicated().sum())
        },
        "columns": columns_profile
    }


def print_dataset_profile(profile):
    """
    Display dataset profile in a readable format.
    """

    dataset = profile["dataset"]

    print("\n========== DATASET PROFILE ==========")

    print(f"Total Rows: {dataset['total_rows']}")
    print(f"Total Columns: {dataset['total_columns']}")
    print(f"Total Cells: {dataset['total_cells']}")
    print(f"Duplicate Rows: {dataset['duplicate_rows']}")

    print("\n========== COLUMN PROFILES ==========")

    for column in profile["columns"]:

        print(f"\nColumn: {column['column']}")
        print(f"Type: {column['data_type']}")
        print(f"Missing: {column['missing_count']}")
        print(f"Missing %: {column['missing_percentage']}")
        print(f"Unique Values: {column['unique_count']}")
        print(f"Samples: {column['sample_values']}")
