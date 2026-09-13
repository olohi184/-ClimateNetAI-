"""Compare historical and reproduced ClimateNetAI model results.

This module does not alter either results table. It aligns historical and
reproduced monthly/model records and reports metric differences transparently.

Usage:
    python src/evaluate_models.py \
        monthly_model_results.csv \
        reproduced_results/reproduced_monthly_model_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

KEYS = ["Month", "Model"]
METRICS = ["MAE", "RMSE", "R2"]


def load_results(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} results not found: {path}")
    df = pd.read_csv(path)
    required = KEYS + METRICS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} results missing columns: {missing}")
    if df.duplicated(KEYS).any():
        dup = df.loc[df.duplicated(KEYS, keep=False), KEYS]
        raise ValueError(f"{label} contains duplicate Month/Model keys:\n{dup}")
    return df.copy()


def compare_results(
    historical_path: str | Path,
    reproduced_path: str | Path,
) -> pd.DataFrame:
    historical = load_results(historical_path, "Historical")
    reproduced = load_results(reproduced_path, "Reproduced")

    keep_hist = historical[KEYS + METRICS].rename(
        columns={m: f"{m}_Historical" for m in METRICS}
    )
    keep_repr_cols = KEYS + METRICS + [
        c for c in ["Validation_Method", "Feature_Mode", "Reproduction_Random_State"]
        if c in reproduced.columns
    ]
    keep_repr = reproduced[keep_repr_cols].rename(
        columns={m: f"{m}_Reproduced" for m in METRICS}
    )

    merged = keep_hist.merge(keep_repr, on=KEYS, how="outer", indicator=True)

    for metric in METRICS:
        merged[f"{metric}_Difference"] = (
            merged[f"{metric}_Reproduced"] - merged[f"{metric}_Historical"]
        )
        merged[f"{metric}_Absolute_Difference"] = merged[f"{metric}_Difference"].abs()

    merged["Record_Status"] = merged["_merge"].map(
        {"both": "Matched", "left_only": "Historical only", "right_only": "Reproduced only"}
    )
    return merged.drop(columns=["_merge"])


def best_model_by_month(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Select best model primarily by highest R², then lower RMSE/MAE."""
    r2 = f"R2_{suffix}"
    rmse = f"RMSE_{suffix}"
    mae = f"MAE_{suffix}"
    valid = df.dropna(subset=[r2]).copy()
    valid = valid.sort_values(
        ["Month", r2, rmse, mae],
        ascending=[True, False, True, True],
    )
    best = valid.groupby("Month", as_index=False).first()
    return best[["Month", "Model"]].rename(columns={"Model": f"Best_{suffix}"})


def build_best_model_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    hist = best_model_by_month(comparison, "Historical")
    rep = best_model_by_month(comparison, "Reproduced")
    out = hist.merge(rep, on="Month", how="outer")
    out["Same_Best_Model"] = out["Best_Historical"] == out["Best_Reproduced"]
    return out


def write_reports(
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "historical_vs_reproduced_results.csv"
    best_path = output_dir / "best_model_comparison.csv"

    comparison.to_csv(comparison_path, index=False)
    best = build_best_model_comparison(comparison)
    best.to_csv(best_path, index=False)
    return comparison_path, best_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare historical and reproduced ClimateNetAI results."
    )
    parser.add_argument("historical_results", help="Historical monthly_model_results.csv")
    parser.add_argument("reproduced_results", help="Reproduced monthly results CSV")
    parser.add_argument(
        "--output-dir",
        default="reproduced_results",
        help="Directory for comparison reports.",
    )
    args = parser.parse_args()

    comparison = compare_results(args.historical_results, args.reproduced_results)
    comparison_path, best_path = write_reports(comparison, args.output_dir)
    best = build_best_model_comparison(comparison)

    matched = int((comparison["Record_Status"] == "Matched").sum())
    same_best = int(best["Same_Best_Model"].fillna(False).sum())

    print("=" * 72)
    print("ClimateNetAI — Historical vs Reproduced Evaluation")
    print("=" * 72)
    print(f"Matched month/model records: {matched}")
    print(f"Months compared: {len(best)}")
    print(f"Same best model: {same_best}/{len(best)}")
    print("\nMean absolute metric differences:")
    for metric in METRICS:
        col = f"{metric}_Absolute_Difference"
        print(f"  {metric}: {comparison[col].mean():.6f}")
    print(f"\nDetailed comparison: {comparison_path}")
    print(f"Best-model comparison: {best_path}")
    print(
        "\nInterpretation: differences are reported, not corrected or hidden. "
        "The reproduction uses a documented deterministic seed and may not "
        "exactly reproduce historical experiments whose original seed/split "
        "was not preserved."
    )


if __name__ == "__main__":
    main()
