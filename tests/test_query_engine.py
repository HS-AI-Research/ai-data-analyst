
"""
Regression tests for the query engine.

Covers:
- Row counting
- Distinct counting
- Business metrics
- Grouped aggregations
- Sales and HR datasets
"""

import pytest

from src.data_loader import load_data
from src.query_engine import execute_query


@pytest.fixture(scope="module")
def sales_df():
    return load_data("data/raw/orders.csv")


@pytest.fixture(scope="module")
def hr_df():
    return load_data("data/raw/employees.csv")


# --------------------------------------------------
# Row and column counting
# --------------------------------------------------

def test_sales_row_count(sales_df):
    result = execute_query(
        sales_df,
        {"operation": "count", "column": None},
    )

    assert result["count"] == 35


def test_hr_row_count(hr_df):
    result = execute_query(
        hr_df,
        {"operation": "count", "column": None},
    )

    assert result["count"] == 10


def test_distinct_products(sales_df):
    result = execute_query(
        sales_df,
        {
            "operation": "count",
            "column": "product",
            "aggregation": "nunique",
        },
    )

    assert result["count"] == 6


def test_distinct_cities(sales_df):
    result = execute_query(
        sales_df,
        {
            "operation": "count",
            "column": "city",
            "aggregation": "nunique",
        },
    )

    assert result["count"] == 5


def test_distinct_departments(hr_df):
    result = execute_query(
        hr_df,
        {
            "operation": "count",
            "column": "department",
            "aggregation": "nunique",
        },
    )

    assert result["count"] == 4


# --------------------------------------------------
# Sales business metrics
# --------------------------------------------------

def test_sales_orders_count(sales_df):
    result = execute_query(
        sales_df,
        {
            "operation": "count",
            "metric": "orders",
        },
    )

    assert result["count"] == 35


def test_sales_orders_grouped_by_city(sales_df):
    result = execute_query(
        sales_df,
        {
            "operation": "groupby",
            "metric": "orders",
            "group_by": "city",
            "aggregation": "nunique",
        },
    )

    actual = {
        item["city"]: item["result"]
        for item in result["results"]
    }

    expected = {
        "Tehran": 13,
        "Isfahan": 6,
        "Shiraz": 6,
        "Mashhad": 5,
        "Tabriz": 5,
    }

    assert actual == expected


# --------------------------------------------------
# HR business metrics
# --------------------------------------------------

def test_employee_count(hr_df):
    result = execute_query(
        hr_df,
        {
            "operation": "count",
            "metric": "employee_count",
        },
    )

    assert result["count"] == 10


def test_employee_count_grouped_by_department(hr_df):
    result = execute_query(
        hr_df,
        {
            "operation": "groupby",
            "metric": "employee_count",
            "group_by": "department",
            "aggregation": "nunique",
        },
    )

    actual = {
        item["department"]: item["result"]
        for item in result["results"]
    }

    expected = {
        "Sales": 3,
        "IT": 3,
        "HR": 2,
        "Finance": 2,
    }

    assert actual == expected


# --------------------------------------------------
# HR numeric aggregations
# --------------------------------------------------

def test_average_salary(hr_df):
    result = execute_query(
        hr_df,
        {
            "operation": "mean",
            "metric": "average_salary",
        },
    )

    assert result["result"] == pytest.approx(4930.0)


def test_total_salary_by_department(hr_df):
    result = execute_query(
        hr_df,
        {
            "operation": "groupby",
            "metric": "total_salary",
            "group_by": "department",
            "aggregation": "sum",
        },
    )

    actual = {
        item["department"]: item["result"]
        for item in result["results"]
    }

    expected = {
        "Sales": 14800,
        "IT": 14200,
        "Finance": 12400,
        "HR": 7900,
    }

    assert actual == expected
