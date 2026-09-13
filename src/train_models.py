"""Train reproduced ClimateNetAI monthly machine-learning models.

This script creates a reproducible reconstruction of the documented
ClimateNetAI monthly modelling workflow. It DOES NOT overwrite the
historical archived models in ``models/monthly_models``.

Important reproducibility note
------------------------------
The original thesis experiments did not clearly document a fixed random
seed. This implementation therefore uses ``random_state=42`` as a NEW
reproduction seed so repeated runs of this script are deterministic.
It must not be described as the original experimental seed.

Default workflow
----------------
- Predictors: Temperature, Pressure, Relative_Humidity
- Target: RSSI
- Models: Linear Regression, Decision Tree, Random Forest, XGBoost
- Non-September months: 80/20 holdout for reconstructed validation
- September: Leave-One-Out Cross Validation (LOOCV) because n=8
- After validation, each model is refit on all observations for that month
  and saved to ``models/reproduced_monthly_models``.

Usage
-----
From the repository root:

    python src/train_models.py modeling_dataset_v1.csv

Optional four-feature mode:

    python src/train_models.py modeling_dataset_v1.csv --features 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Allow direct execution from repository root or src/.
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (  # noqa: E402
    get_feature_columns,
    get_month_data,
    partition_by_month,
    prepare_dataset,
)

REPRODUCTION_RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "Linear Regression": {
        "class": "LinearRegression",
        "parameters": {},
    },
    "Decision Tree": {
        "class": "DecisionTreeRegressor",
        "parameters": {
            "max_depth": 10,
            "criterion": "squared_error",
            "random_state": REPRODUCTION_RANDOM_STATE,
        },
    },
    "Random Forest": {
        "class": "RandomForestRegressor",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": REPRODUCTION_RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "XGBoost": {
        "class": "XGBRegressor",
        "parameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "objective": "reg:squarederror",
            "random_state": REPRODUCTION_RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": 0,
        },
    },
}


def build_models() -> dict[str, Any]:
    """Instantiate the four reconstructed ClimateNetAI regressors."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            criterion="squared_error",
            random_state=REPRODUCTION_RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=REPRODUCTION_RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            objective="reg:squarederror",
            random_state=REPRODUCTION_RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE and R²."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def validate_holdout(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Reconstructed 80/20 holdout validation for ordinary months."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=REPRODUCTION_RANDOM_STATE,
    )

    fitted = clone(model)
    start = time.perf_counter()
    fitted.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = fitted.predict(X_test)
    prediction_time = time.perf_counter() - start

    metrics = regression_metrics(y_test.to_numpy(), np.asarray(pred))
    metrics.update(
        {
            "Validation_Method": "80/20 holdout (reproduction)",
            "Train_N": int(len(X_train)),
            "Test_N": int(len(X_test)),
            "Training_Time": float(training_time),
            "Prediction_Time": float(prediction_time),
        }
    )
    return metrics


def validate_loocv(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """LOOCV reconstruction for September's eight observations."""
    loo = LeaveOneOut()
    y_true: list[float] = []
    y_pred: list[float] = []
    total_training_time = 0.0
    total_prediction_time = 0.0

    for train_idx, test_idx in loo.split(X):
        fitted = clone(model)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        start = time.perf_counter()
        fitted.fit(X_train, y_train)
        total_training_time += time.perf_counter() - start

        start = time.perf_counter()
        pred = fitted.predict(X_test)
        total_prediction_time += time.perf_counter() - start

        y_true.append(float(y_test.iloc[0]))
        y_pred.append(float(pred[0]))

    metrics = regression_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics.update(
        {
            "Validation_Method": "LOOCV (reconstructed September validation)",
            "Train_N": int(len(X) - 1),
            "Test_N": 1,
            "Training_Time": float(total_training_time),
            "Prediction_Time": float(total_prediction_time),
        }
    )
    return metrics


def safe_model_filename(month: str, model_name: str) -> str:
    """Return a stable filename for a reproduced model artifact."""
    slug = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{month.lower()}_{slug}.pkl"


def train_monthly_models(
    csv_path: str | Path,
    feature_mode: int,
    output_root: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """Validate, train, evaluate, refit and save reproduced monthly models."""
    df = prepare_dataset(csv_path)
    partitions = partition_by_month(df)

    output_root = Path(output_root)
    model_dir = output_root / "models" / "reproduced_monthly_models"
    results_dir = output_root / "reproduced_results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    model_templates = build_models()

    for month, month_df in partitions.items():
        X, y = get_month_data(df, month, feature_mode)

        for model_name, template in model_templates.items():
            if month.lower() == "september" and len(X) <= 10:
                metrics = validate_loocv(template, X, y)
            else:
                metrics = validate_holdout(template, X, y)

            # Deployment/research artifact: fit on every available observation
            # after validation. Historical models are never overwritten.
            final_model = clone(template)
            final_model.fit(X, y)

            artifact_path = model_dir / safe_model_filename(month, model_name)
            joblib.dump(final_model, artifact_path)

            results.append(
                {
                    "Month": month,
                    "Model": model_name,
                    "N": int(len(X)),
                    "Feature_Mode": int(feature_mode),
                    "Features": "|".join(get_feature_columns(feature_mode)),
                    **metrics,
                    "Reproduction_Random_State": REPRODUCTION_RANDOM_STATE,
                    "Model_File": str(artifact_path.relative_to(output_root)),
                }
            )

    results_df = pd.DataFrame(results)
    results_path = results_dir / "reproduced_monthly_model_results.csv"
    results_df.to_csv(results_path, index=False)

    metadata = {
        "status": "reconstructed reproducibility run",
        "source_dataset": str(csv_path),
        "feature_mode": feature_mode,
        "features": get_feature_columns(feature_mode),
        "target": "RSSI",
        "reproduction_random_state": REPRODUCTION_RANDOM_STATE,
        "important_note": (
            "random_state=42 is a reproduction seed chosen for deterministic "
            "reruns; it is not claimed to be the original thesis seed."
        ),
        "ordinary_validation": "80/20 holdout",
        "september_validation": "LOOCV when September n <= 10",
        "model_config": MODEL_CONFIG,
        "historical_models_overwritten": False,
    }
    metadata_path = results_dir / "reproduction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return results_df, results_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train reproduced ClimateNetAI monthly models."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="modeling_dataset_v1.csv",
        help="Path to modeling_dataset_v1.csv",
    )
    parser.add_argument(
        "--features",
        type=int,
        choices=[3, 4],
        default=3,
        help="Feature mode: 3 atmospheric predictors or 4 including Month_Number.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Repository/output root. Default: current directory.",
    )
    args = parser.parse_args()

    results, results_path = train_monthly_models(
        args.csv_path,
        args.features,
        args.output_root,
    )

    print("=" * 72)
    print("ClimateNetAI — Reproduced Monthly Model Training")
    print("=" * 72)
    print(f"Models evaluated: {len(results)}")
    print(f"Months: {results['Month'].nunique()}")
    print(f"Model families: {results['Model'].nunique()}")
    print(f"Feature mode: {args.features}")
    print(
        "Reproduction seed: 42 "
        "(chosen for deterministic reruns; NOT claimed as original thesis seed)"
    )
    print(f"Results: {results_path}")
    print(
        f"Saved reproduced models under: "
        f"{Path(args.output_root) / 'models' / 'reproduced_monthly_models'}"
    )
    print("\nValidation methods:")
    print(results.groupby("Validation_Method").size().to_string())
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
