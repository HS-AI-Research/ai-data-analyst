
import pandas as pd
import pytest

from src.eda_engine import (
    analyze_dataset,
    build_llm_evidence,
    is_exploratory_request,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "record_id": range(1, 13),
        "event_date": [
            f"2026-01-{day:02d}"
            for day in range(1, 13)
        ],
        "department": [
            "Sales", "Sales", "IT", "IT",
            "HR", "HR", "Sales", "IT",
            "HR", "Sales", "IT", "HR",
        ],
        "amount": [
            10, 12, 11, 13, 12, 14,
            15, 16, 14, 13, 12, 500,
        ],
        "score": [
            4.0, 4.2, 3.8, 4.1,
            4.5, 4.0, 3.9, 4.3,
            4.1, 4.4, 4.2, None,
        ],
    })


def test_eda_returns_expected_sections(sample_df):
    result = analyze_dataset(sample_df)

    expected = {
        "analysis_type",
        "dataset",
        "numeric",
        "categorical",
        "dates",
        "outliers",
        "identifiers",
        "correlations",
        "quality_warnings",
        "limitations",
    }

    assert expected.issubset(result.keys())


def test_dataset_shape_is_correct(sample_df):
    result = analyze_dataset(sample_df)

    assert result["dataset"]["rows"] == 12
    assert result["dataset"]["columns"] == 5


def test_missing_values_are_detected(sample_df):
    result = analyze_dataset(sample_df)

    assert result["dataset"]["missing_cells"] == 1
    assert "score" in result["dataset"]["missing_by_column"]


def test_date_column_is_detected(sample_df):
    result = analyze_dataset(sample_df)

    assert "event_date" in result["dates"]
    assert result["dates"]["event_date"]["valid_count"] == 12


def test_identifier_is_excluded_from_numeric_profile(
    sample_df,
):
    result = analyze_dataset(sample_df)

    assert "record_id" not in result["numeric"]
    assert "record_id" in result["identifiers"]


def test_identifier_is_excluded_from_correlations(
    sample_df,
):
    result = analyze_dataset(sample_df)

    for pair in result["correlations"]["pairs"]:
        assert pair["column_1"] != "record_id"
        assert pair["column_2"] != "record_id"


def test_potential_outlier_is_flagged(sample_df):
    result = analyze_dataset(sample_df)

    assert "amount" in result["outliers"]
    assert result["outliers"]["amount"]["count"] >= 1


def test_source_dataframe_is_not_modified(sample_df):
    original = sample_df.copy(deep=True)

    analyze_dataset(sample_df)

    pd.testing.assert_frame_equal(
        sample_df,
        original,
    )


def test_llm_evidence_does_not_include_raw_rows(sample_df):
    evidence = analyze_dataset(sample_df)
    compact = build_llm_evidence(evidence)

    assert "rows_data" not in compact
    assert "sample_rows" not in compact
    assert "record_id" not in compact["numeric"]


def test_exploratory_request_detection():
    assert is_exploratory_request(
        "What are the most important business insights "
        "and potential risks in this dataset?"
    )

    assert is_exploratory_request(
        "Perform exploratory data analysis"
    )

    assert is_exploratory_request(
        "مهم‌ترین بینش‌ها و ریسک‌های این داده چیست؟"
    )


def test_ranking_question_is_not_exploratory():
    assert not is_exploratory_request(
        "Which products have the most orders?"
    )


def test_specific_count_question_is_not_exploratory():
    assert not is_exploratory_request(
        "What is the total number of orders?"
    )


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError):
        analyze_dataset(pd.DataFrame())


def test_non_dataframe_is_rejected():
    with pytest.raises(TypeError):
        analyze_dataset([1, 2, 3])
