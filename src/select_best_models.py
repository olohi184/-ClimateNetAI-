"""Select the best reproduced ClimateNetAI model for each month.

Ranks candidate models primarily by highest R², then lower RMSE and MAE.
The output is a NEW reproducibility artifact and does not overwrite the
historical best_model_per_month_validated.csv.

Usage:
    python src/select_best_models.py \
        reproduced_results/reproduced_monthly_model_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED = ["Month", "Model", "MAE", "RMSE", "R2"]


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Results file missing required columns: {missing}")

    if df.duplicated(["Month", "Model"]).any():
        raise ValueError("Duplicate Month/Model records detected.")

    return df.copy()


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """Choose one model per month using transparent deterministic ranking."""
    ranked = df.sort_values(
        ["Month", "R2", "RMSE", "MAE", "Model"],
        ascending=[True, False, True, True, True],
    ).copy()

    best = ranked.groupby("Month", as_index=False).first()

    keep = ["Month", "Model", "MAE", "RMSE", "R2"]
    optional = [
        "N",
        "Feature_Mode",
        "Validation_Method",
        "Reproduction_Random_State",
        "Model_File",
    ]
    keep += [c for c in optional if c in best.columns]
    best = best[keep].copy()

    best["Selection_Rule"] = (
        "Highest R2; ties resolved by lower RMSE, lower MAE, then model name"
    )
    best["Recommendation_Status"] = "Reproduced best model"

    # Calendar order when standard month names are present.
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    best["_month_order"] = best["Month"].map(month_order).fillna(99)
    best = best.sort_values("_month_order").drop(columns="_month_order").reset_index(drop=True)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best reproduced ClimateNetAI model per month."
    )
    parser.add_argument("results_csv", help="Reproduced monthly model results CSV.")
    parser.add_argument(
        "--output",
        default="reproduced_results/reproduced_best_model_per_month.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = load_results(args.results_csv)
    best = select_best_models(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output, index=False)

    print("=" * 72)
    print("ClimateNetAI — Reproduced Best-Model Selection")
    print("=" * 72)
    print(f"Months selected: {len(best)}")
    print("Selection rule: highest R², then lower RMSE, lower MAE.")
    print("\nSelected models:")
    print(best[["Month", "Model", "R2"]].to_string(index=False))
    print(f"\nOutput: {output}")
    print(
        "\nHistorical best_model_per_month_validated.csv was not modified."
    )


if __name__ == "__main__":
    main()
