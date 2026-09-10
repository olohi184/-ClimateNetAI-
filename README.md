# ClimateNetAI

## Live Application

ClimateNetAI v1.0.1 is publicly available at:

https://climatenetai.streamlit.app


ClimateNetAI is a climate-aware machine-learning research application for predicting wireless signal strength under atmospheric variability.

## Version

**ClimateNetAI v1.0.1**

Research Prototype  
Developed by **Olohimai Juliet Michael**

## Purpose

ClimateNetAI investigates how environmental conditions influence wireless signal behavior and provides model-based RSSI predictions using atmospheric and temporal inputs.

The application is designed to support research into climate-resilient digital connectivity and machine-learning-assisted wireless network analysis.

## Input Variables

ClimateNetAI currently supports the following prediction inputs:

- Temperature
- Atmospheric Pressure
- Relative Humidity
- Month

Depending on the trained monthly model, either three or four features may be used internally.

## Target Variable

The prediction target is:

- **RSSI — Received Signal Strength Indicator (dBm)**

## Machine Learning Models

ClimateNetAI currently supports:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

Model availability may vary by month where validation or reliability checks indicate that a model should not be exposed for prediction.

## Application Features

ClimateNetAI v1.0.1 includes:

- Monthly RSSI prediction
- Signal-quality classification
- Practical prediction interpretation
- MAE, RMSE, and R² validation metrics
- Model reliability notices
- Monthly best-model recommendation
- Side-by-side comparison of all monthly models
- Monthly R² performance trend visualization
- Research and data-quality notes
- Downloadable prediction reports
- Month-aware model availability controls

## Model Validation

The application reports:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Model recommendations are based primarily on validation R² while also displaying MAE and RMSE for context.

Negative R² values indicate poor out-of-sample generalization relative to a mean-prediction baseline.

## Research and Data-Quality Notes

### June and July

The current modeling dataset contains identical Temperature, Pressure, Relative Humidity, and RSSI observations for June and July.

These months should therefore not be treated as independent monthly evidence until the original source data is verified.

### September

September contains 8 observations in the current modeling dataset.

Its MAE, RMSE, and R² values were reconstructed using Leave-One-Out Cross Validation (LOOCV) with:

- Temperature
- Pressure
- Relative Humidity
- Month Number

## Running ClimateNetAI Locally

From the project directory:

```bash
python3 -m streamlit run app.py
```

## Core Project Files

Important files include:

```text
app.py
modeling_dataset_v1.csv
monthly_model_results.csv
best_model_per_month_validated.csv
models/monthly_models/
requirements.txt
CITATION.cff
LICENSE
```

## License

ClimateNetAI is released under the **MIT License**. See the `LICENSE` file for the full license text.

## Citation

If you use ClimateNetAI in research, please cite the archived software release:

**Michael, Olohimai Juliet. ClimateNetAI: Climate-Aware Machine Learning for Wireless Signal Prediction, version 1.0.1. Zenodo. https://doi.org/10.5281/zenodo.21892444**

## Model Persistence and Reproducibility

The archived ClimateNetAI model set preserves the trained model artefacts used by this research release. Inspection of the serialized estimators indicates that models were created under scikit-learn 1.6.1 and 1.9.0. Cross-version loading of persisted scikit-learn estimators may produce compatibility warnings and is not guaranteed by scikit-learn.

The original model artefacts are intentionally retained rather than retrained or re-serialized solely to remove these warnings, because doing so could change the research artefact. For reproducible inference, users should use the documented dependency environment and interpret any cross-version compatibility warning as an environment limitation rather than evidence that a prediction has been independently reproduced under the model's original training environment.

The release therefore distinguishes model integrity and prediction consistency from cross-version environment compatibility. Future releases should standardize the training and inference environment and archive the corresponding training code and environment metadata alongside the models.
