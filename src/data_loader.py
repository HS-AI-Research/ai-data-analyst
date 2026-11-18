
"""
General-Purpose Data Loader

Responsibilities:
- Load CSV and Excel datasets.
- Detect common CSV encodings and delimiters.
- Support Excel sheet selection.
- Validate loaded datasets.
- Normalize column names safely.
- Preserve original data values.
"""

import csv
import io
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}

CSV_ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "cp1256",
    "windows-1252",
    "latin1",
]

CSV_DELIMITERS = ",;\t|"


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a loaded DataFrame."""

    if df is None:
        raise ValueError("The loaded dataset is None.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    # Normalize column names without changing data values.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if any(not column for column in df.columns):
        raise ValueError(
            "The dataset contains an empty column name."
        )

    if not df.columns.is_unique:
        duplicates = (
            df.columns[df.columns.duplicated()]
            .tolist()
        )

        raise ValueError(
            f"Duplicate column names detected: {duplicates}"
        )

    return df


def validate_csv_header(header: list) -> None:
    """Validate original CSV column names before Pandas modifies them."""

    if not header:
        raise ValueError(
            "The CSV file does not contain column names."
        )

    normalized_header = [
        str(column).strip()
        for column in header
    ]

    if any(not column for column in normalized_header):
        raise ValueError(
            "The dataset contains an empty column name."
        )

    seen = set()
    duplicates = []

    for column in normalized_header:
        if column in seen and column not in duplicates:
            duplicates.append(column)

        seen.add(column)

    if duplicates:
        raise ValueError(
            f"Duplicate column names detected: {duplicates}"
        )


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV using common encodings and delimiter detection."""

    last_error = None

    for encoding in CSV_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)

            if not text.strip():
                raise ValueError(
                    "The dataset is empty."
                )

            sample = text[:8192]

            try:
                dialect = csv.Sniffer().sniff(
                    sample,
                    delimiters=CSV_DELIMITERS,
                )
            except csv.Error:
                # Fallback for single-column CSV files.
                dialect = csv.excel

            reader = csv.reader(
                io.StringIO(text),
                dialect=dialect,
            )

            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(
                    "The dataset is empty."
                )

            validate_csv_header(header)

            return pd.read_csv(
                io.StringIO(text),
                sep=dialect.delimiter,
                engine="python",
            )

        except UnicodeDecodeError as error:
            last_error = error

        except pd.errors.ParserError as error:
            last_error = error

        except csv.Error as error:
            last_error = error

    raise ValueError(
        f"Could not read CSV file: {path.name}. "
        f"Last error: {last_error}"
    )


def load_excel(
    path: Path,
    sheet_name=0,
) -> pd.DataFrame:
    """Load an Excel worksheet."""

    try:
        return pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

    except ImportError as error:
        raise ImportError(
            "Excel support requires openpyxl for .xlsx "
            "or xlrd for .xls files."
        ) from error

    except ValueError as error:
        raise ValueError(
            f"Could not read Excel sheet '{sheet_name}' "
            f"from {path.name}: {error}"
        ) from error


def load_data(
    file_path: str,
    sheet_name=0,
) -> pd.DataFrame:
    """
    Load CSV or Excel files into a validated DataFrame.

    Args:
        file_path: Dataset path.
        sheet_name: Excel sheet name or zero-based index.

    Returns:
        Validated Pandas DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"The specified path is not a file: {path}"
        )

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".csv":
        df = load_csv(path)

    else:
        df = load_excel(path, sheet_name)

    return validate_dataframe(df)
