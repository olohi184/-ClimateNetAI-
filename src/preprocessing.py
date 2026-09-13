"""ClimateNetAI preprocessing utilities.

Prepares validated modelling data for month-wise ML experiments without
modifying the source CSV.

Usage examples:
    python src/preprocessing.py modeling_dataset_v1.csv
    python src/preprocessing.py modeling_dataset_v1.csv --month September --features 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

BASE_FEATURES = ["Temperature", "Pressure", "Relative_Humidity"]
FOUR_FEATURES = ["Temperature", "Pressure", "Relative_Humidity", "Month_Number"]
TARGET = "RSSI"

REQUIRED_COLUMNS = [
    "Temperature",
    "Pressure",
    "Relative_Humidity",
    "RSSI",
    "Month",
    "Month_Number",
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the ClimateNetAI modelling dataset from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df.copy()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert modelling columns to numeric and fail if coercion creates NaNs."""
    out = df.copy()
    numeric_cols = [
        "Temperature",
        "Pressure",
        "Relative_Humidity",
        "RSSI",
        "Month_Number",
    ]

    for col in numeric_cols:
        before_missing = out[col].isna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        after_missing = out[col].isna().sum()

        if after_missing > before_missing:
            raise ValueError(
                f"Column '{col}' contains non-numeric value(s) that could not be converted."
            )

    return out


def check_missing_values(df: pd.DataFrame) -> None:
    """Raise an error if required modelling fields contain missing values.

    The current ClimateNetAI modelling dataset contains no missing values.
    This function intentionally does not impute or interpolate data silently.
    """
    missing = df[REQUIRED_COLUMNS].isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            "Missing values detected in required modelling fields: "
            f"{missing.to_dict()}. "
            "No automatic interpolation/imputation is applied by this pipeline."
        )


def get_feature_columns(feature_mode: Literal[3, 4] | int = 3) -> list[str]:
    """Return the documented ClimateNetAI feature set."""
    if feature_mode == 3:
        return BASE_FEATURES.copy()
    if feature_mode == 4:
        return FOUR_FEATURES.copy()
    raise ValueError("feature_mode must be 3 or 4.")


def prepare_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load, validate numeric types, and check missing values."""
    df = load_dataset(csv_path)
    df = coerce_numeric(df)
    check_missing_values(df)
    return df


def partition_by_month(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return month-specific dataframes in Month_Number order."""
    ordered = df.sort_values(["Month_Number"], kind="stable").copy()
    partitions: dict[str, pd.DataFrame] = {}

    for month_no in sorted(ordered["Month_Number"].dropna().astype(int).unique()):
        month_df = ordered[ordered["Month_Number"].astype(int) == month_no].copy()
        month_name = str(month_df["Month"].iloc[0])
        partitions[month_name] = month_df.reset_index(drop=True)

    return partitions


def get_month_data(
    df: pd.DataFrame,
    month: str,
    feature_mode: Literal[3, 4] | int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y for one calendar month."""
    matches = df[df["Month"].astype(str).str.lower() == month.lower()].copy()

    if matches.empty:
        available = ", ".join(partition_by_month(df).keys())
        raise ValueError(
            f"Month '{month}' not found. Available months: {available}"
        )

    features = get_feature_columns(feature_mode)
    X = matches[features].reset_index(drop=True)
    y = matches[TARGET].reset_index(drop=True)

    return X, y


def describe_preprocessing(
    df: pd.DataFrame,
    feature_mode: Literal[3, 4] | int = 3,
) -> None:
    """Print a concise reproducibility summary."""
    features = get_feature_columns(feature_mode)
    partitions = partition_by_month(df)

    print("=" * 68)
    print("ClimateNetAI — Preprocessing Summary")
    print("=" * 68)
    print(f"Rows available: {len(df)}")
    print(f"Feature mode: {feature_mode}-feature")
    print(f"Predictors: {', '.join(features)}")
    print(f"Target: {TARGET}")
    print("Missing-data policy: fail explicitly; no silent imputation.")
    print("Scaling policy: none applied in this preprocessing layer.")
    print("Outlier policy: none removed in this preprocessing layer.")
    print("\nMonth partitions:")
    for month, month_df in partitions.items():
        print(f"  {month}: {len(month_df)} observations")

    if "September" in partitions and len(partitions["September"]) <= 10:
        print(
            "\n[NOTE] September is a small-sample month. "
            "Validation strategy must preserve the documented special treatment."
        )

    if "June" in partitions and "July" in partitions:
        cols = ["Temperature", "Pressure", "Relative_Humidity", "RSSI"]
        june = partitions["June"][cols].reset_index(drop=True)
        july = partitions["July"][cols].reset_index(drop=True)

        if june.equals(july):
            print(
                "[NOTE] June and July modelling observations are identical. "
                "Do not treat them as independent evidence until source data are verified."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ClimateNetAI modelling data reproducibly."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="modeling_dataset_v1.csv",
        help="Path to modelling dataset.",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Optional month to extract, e.g. September.",
    )
    parser.add_argument(
        "--features",
        type=int,
        choices=[3, 4],
        default=3,
        help="Use 3 environmental predictors or 4 predictors including Month_Number.",
    )
    args = parser.parse_args()

    df = prepare_dataset(args.csv_path)
    describe_preprocessing(df, args.features)

    if args.month:
        X, y = get_month_data(df, args.month, args.features)
        print("\nRequested month dataset:")
        print(f"  Month: {args.month}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Columns: {list(X.columns)}")


if __name__ == "__main__":
    main()
