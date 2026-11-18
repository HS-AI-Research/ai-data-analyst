
import pandas as pd
import pytest

from src.data_loader import load_data


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "employee_id": [101, 102, 103],
        "name": ["Ali", "Sara", "Reza"],
        "department": ["IT", "HR", "Sales"],
        "salary": [5000, 4500, 6000],
    })


def test_load_csv(tmp_path, sample_dataframe):
    file_path = tmp_path / "employees.csv"

    sample_dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    df = load_data(file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        "employee_id",
        "name",
        "department",
        "salary",
    ]
    assert df["salary"].tolist() == [5000, 4500, 6000]


def test_load_excel(tmp_path, sample_dataframe):
    file_path = tmp_path / "employees.xlsx"

    sample_dataframe.to_excel(
        file_path,
        index=False
    )

    df = load_data(file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        "employee_id",
        "name",
        "department",
        "salary",
    ]
    assert df["salary"].tolist() == [5000, 4500, 6000]


def test_load_csv_preserves_persian_text(tmp_path):
    file_path = tmp_path / "persian_data.csv"

    expected_names = ["علی", "سارا", "رضا"]

    pd.DataFrame({
        "name": expected_names,
        "city": ["تهران", "تبریز", "اصفهان"],
    }).to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    df = load_data(file_path)

    assert df["name"].tolist() == expected_names
    assert df["city"].tolist() == ["تهران", "تبریز", "اصفهان"]


def test_load_empty_csv_raises_error(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_data(file_path)


def test_load_csv_with_empty_column_name_raises_error(tmp_path):
    file_path = tmp_path / "empty_column.csv"
    file_path.write_text("id,,salary\n1,Ali,5000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty column name"):
        load_data(file_path)


def test_load_csv_with_duplicate_columns_raises_error(tmp_path):
    file_path = tmp_path / "duplicate.csv"
    file_path.write_text(
        "id,salary,salary\n1,5000,6000\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate column"):
        load_data(file_path)


def test_load_csv_normalizes_column_names(tmp_path):
    file_path = tmp_path / "spaces.csv"
    file_path.write_text(
        " id , name , salary \n1,Ali,5000\n",
        encoding="utf-8"
    )

    df = load_data(file_path)

    assert list(df.columns) == ["id", "name", "salary"]


def test_unsupported_file_format_raises_error(tmp_path):
    file_path = tmp_path / "data.json"
    file_path.write_text('{"id": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(file_path)


def test_missing_file_raises_error(tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="File not found"):
        load_data(file_path)


def test_directory_path_raises_error(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        load_data(tmp_path)
