
"""
Integration regression tests for the general-purpose assistant.

Covers:
- Persian questions
- Sales dataset
- HR dataset
- Business metrics
- Distinct counts
- Grouped analysis
- Final answer generation
"""

import pytest

from src.data_loader import load_data
from src.assistant import answer_question


@pytest.fixture(scope="module")
def sales_df():
    return load_data("data/raw/orders.csv")


@pytest.fixture(scope="module")
def hr_df():
    return load_data("data/raw/employees.csv")


def run_question(df, question):
    result = answer_question(df, question)

    assert isinstance(result, dict)
    assert "evidence" in result
    assert "answer" in result
    assert result["evidence"] is not None
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()

    return result


# --------------------------------------------------
# Sales tests
# --------------------------------------------------

def test_sales_order_count(sales_df):
    result = run_question(
        sales_df,
        "تعداد سفارش‌ها چند است؟",
    )

    assert result["evidence"]["count"] == 35
    assert "35" in result["answer"]


def test_sales_distinct_products(sales_df):
    result = run_question(
        sales_df,
        "تعداد محصولات متفاوت چند است؟",
    )

    assert result["evidence"]["count"] == 6
    assert "6" in result["answer"]


def test_sales_distinct_cities(sales_df):
    result = run_question(
        sales_df,
        "تعداد شهرهای متفاوت در داده‌ها چند است؟",
    )

    assert result["evidence"]["count"] == 5
    assert "5" in result["answer"]


def test_sales_orders_by_city(sales_df):
    result = run_question(
        sales_df,
        "تعداد سفارش‌ها به تفکیک شهر چقدر است؟",
    )

    actual = {
        item["city"]: item["result"]
        for item in result["evidence"]["results"]
    }

    assert actual == {
        "Tehran": 13,
        "Isfahan": 6,
        "Shiraz": 6,
        "Mashhad": 5,
        "Tabriz": 5,
    }


# --------------------------------------------------
# HR tests
# --------------------------------------------------

def test_employee_count(hr_df):
    result = run_question(
        hr_df,
        "تعداد کارکنان چند است؟",
    )

    assert result["evidence"]["count"] == 10
    assert "10" in result["answer"]


def test_average_salary(hr_df):
    result = run_question(
        hr_df,
        "میانگین حقوق کارکنان چقدر است؟",
    )

    assert result["evidence"]["result"] == pytest.approx(4930.0)
    assert "4,930" in result["answer"]


def test_employee_count_by_department(hr_df):
    result = run_question(
        hr_df,
        "تعداد کارکنان به تفکیک دپارتمان چقدر است؟",
    )

    actual = {
        item["department"]: item["result"]
        for item in result["evidence"]["results"]
    }

    assert actual == {
        "Sales": 3,
        "IT": 3,
        "HR": 2,
        "Finance": 2,
    }


def test_total_salary_by_department(hr_df):
    result = run_question(
        hr_df,
        "مجموع حقوق هر دپارتمان چقدر است؟",
    )

    actual = {
        item["department"]: item["result"]
        for item in result["evidence"]["results"]
    }

    assert actual == {
        "Sales": 14800,
        "IT": 14200,
        "Finance": 12400,
        "HR": 7900,
    }

# --------------------------------------------------
# English tests
# --------------------------------------------------

def test_english_sales_order_count(sales_df):
    result = run_question(
        sales_df,
        "How many orders are there?",
    )

    assert result["evidence"]["count"] == 35
    assert "35" in result["answer"]


def test_english_sales_distinct_products(sales_df):
    result = run_question(
        sales_df,
        "How many distinct products are in the dataset?",
    )

    assert result["evidence"]["count"] == 6
    assert "6" in result["answer"]


def test_english_sales_orders_by_city(sales_df):
    result = run_question(
        sales_df,
        "How many orders are there in each city?",
    )

    actual = {
        item["city"]: item["result"]
        for item in result["evidence"]["results"]
    }

    assert actual == {
        "Tehran": 13,
        "Isfahan": 6,
        "Shiraz": 6,
        "Mashhad": 5,
        "Tabriz": 5,
    }


def test_english_employee_count(hr_df):
    result = run_question(
        hr_df,
        "How many employees are there?",
    )

    assert result["evidence"]["count"] == 10
    assert "10" in result["answer"]


def test_english_average_salary(hr_df):
    result = run_question(
        hr_df,
        "What is the average salary of employees?",
    )

    assert result["evidence"]["result"] == pytest.approx(4930.0)
    assert "4,930" in result["answer"]


def test_english_employee_count_by_department(hr_df):
    result = run_question(
        hr_df,
        "How many employees are in each department?",
    )

    actual = {
        item["department"]: item["result"]
        for item in result["evidence"]["results"]
    }

    assert actual == {
        "Sales": 3,
        "IT": 3,
        "HR": 2,
        "Finance": 2,
    }