
"""
Regression tests for the intent parser.

Covers:
- Persian metric detection
- Distinct count detection
- Business metric resolution
- Grouping detection
- Aggregation detection
"""

import pytest

from src.data_loader import load_data
from src.intent_parser import parse_question


@pytest.fixture(scope="module")
def sales_df():
    return load_data("data/raw/orders.csv")


@pytest.fixture(scope="module")
def hr_df():
    return load_data("data/raw/employees.csv")


# --------------------------------------------------
# Sales: distinct count
# --------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_column",
    [
        ("تعداد محصولات متفاوت چند است؟", "product"),
        ("تعداد شهرهای متفاوت چند است؟", "city"),
        ("تعداد دسته‌بندی‌های متفاوت چند است؟", "category"),
        ("تعداد روش‌های پرداخت متفاوت چند است؟", "payment_method"),
    ],
)
def test_sales_distinct_count(
    sales_df,
    question,
    expected_column,
):
    query = parse_question(question, sales_df)

    assert query["operation"] == "count"
    assert query["column"] == expected_column
    assert query["aggregation"] == "nunique"


# --------------------------------------------------
# HR: distinct count
# --------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_column",
    [
        ("چند دپارتمان متفاوت داریم؟", "department"),
        ("چند جنسیت متفاوت داریم؟", "gender"),
    ],
)
def test_hr_distinct_count(
    hr_df,
    question,
    expected_column,
):
    query = parse_question(question, hr_df)

    assert query["operation"] == "count"
    assert query["column"] == expected_column
    assert query["aggregation"] == "nunique"


# --------------------------------------------------
# Business metrics
# --------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_metric, expected_column",
    [
        ("تعداد سفارش‌ها چند است؟", "orders", "order_id"),
        ("تعداد کارکنان چند است؟", "employee_count", "employee_id"),
    ],
)
def test_business_count_metrics(
    question,
    expected_metric,
    expected_column,
    sales_df,
    hr_df,
):
    df = sales_df if expected_metric == "orders" else hr_df

    query = parse_question(question, df)

    assert query["operation"] == "count"
    assert query["metric"] == expected_metric
    assert query["column"] == expected_column
    assert query["aggregation"] == "nunique"


# --------------------------------------------------
# Grouped business metrics
# --------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_metric, expected_group",
    [
        (
            "تعداد سفارش‌ها به تفکیک شهر چقدر است؟",
            "orders",
            "city",
        ),
        (
            "تعداد کارکنان به تفکیک دپارتمان چقدر است؟",
            "employee_count",
            "department",
        ),
    ],
)
def test_grouped_business_metrics(
    question,
    expected_metric,
    expected_group,
    sales_df,
    hr_df,
):
    df = sales_df if expected_metric == "orders" else hr_df

    query = parse_question(question, df)

    assert query["operation"] == "groupby"
    assert query["metric"] == expected_metric
    assert query["group_by"] == expected_group
    assert query["aggregation"] == "nunique"


# --------------------------------------------------
# HR numeric metrics
# --------------------------------------------------

def test_average_salary_metric(hr_df):
    query = parse_question(
        "میانگین حقوق کارکنان چقدر است؟",
        hr_df,
    )

    assert query["operation"] == "mean"
    assert query["metric"] == "average_salary"


def test_total_salary_grouped_by_department(hr_df):
    query = parse_question(
        "مجموع حقوق هر دپارتمان چقدر است؟",
        hr_df,
    )

    assert query["operation"] == "groupby"
    assert query["metric"] == "total_salary"
    assert query["group_by"] == "department"
    assert query["aggregation"] == "sum"


# --------------------------------------------------
# Sales numeric metrics
# --------------------------------------------------

def test_average_product_price(sales_df):
    query = parse_question(
        "میانگین قیمت هر محصول چقدر است؟",
        sales_df,
    )

    assert query["operation"] == "groupby"
    assert query["metric"] == "average_price"
    assert query["group_by"] == "product"
    assert query["aggregation"] == "mean"

# --------------------------------------------------
# Semantic validation: reject unrelated metrics
# --------------------------------------------------

def test_reject_unavailable_metric_in_sales_dataset(sales_df):
    with pytest.raises(ValueError):
        parse_question(
            "میانگین حقوق کارکنان چقدر است؟",
            sales_df,
        )


def test_accept_available_metric_in_sales_dataset(sales_df):
    query = parse_question(
        "میانگین امتیاز مشتریان چقدر است؟",
        sales_df,
    )

    assert query["operation"] == "mean"
    assert query["metric"] == "average_rating"