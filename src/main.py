
import json
import sys
from pathlib import Path

from src.data_loader import load_data
from src.data_analyzer import analyze_data
from src.advanced_analyzer import analyze_advanced

def main():

    print("=" * 50)
    print("LOCAL AI DATA ANALYST")
    print("=" * 50)

    # Input file
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/orders.csv"
    )

    print(f"\nLoading dataset: {file_path}")

    # Load data
    df = load_data(file_path)

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Data profiling
    print("\nAnalyzing dataset...")

    profile = analyze_data(df)
    # Advanced analysis
    advanced = analyze_advanced(df)

    profile["advanced_analysis"] = advanced

    # Save profile
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "data_profile.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            profile,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print("\nAnalysis completed!")

    print("\nDataset Information:")
    print(json.dumps(
        profile["dataset_info"],
        indent=4,
        ensure_ascii=False
    ))

    print(f"\nProfile saved to: {output_file}")


if __name__ == "__main__":
    main()
