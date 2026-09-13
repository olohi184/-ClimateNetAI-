# ClimateNetAI

**Climate-Aware Machine Learning for Wireless Signal Prediction**

[![Version](https://img.shields.io/badge/version-1.0.3-blue.svg)]("https://github.com/olohi184/ClimateNetAI")
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Software](https://img.shields.io/badge/status-research%20prototype-orange.svg)](https://climatenetai.streamlit.app)

**Founder, Project Originator and Lead Developer:** Olohimai Juliet Michael  
**Current Version:** 1.0.3  
**Status:** Research Prototype

---

## Live Application

ClimateNetAI v1.0.3 is publicly available at:

**https://climatenetai.streamlit.app**

ClimateNetAI is a climate-aware machine-learning research application for predicting wireless signal strength under atmospheric variability.

The project investigates how environmental conditions influence wireless signal behaviour and demonstrates how machine-learning models can support the analysis and prediction of Received Signal Strength Indicator (RSSI).

---

## Research Status

ClimateNetAI is an evolving research software platform informed by ongoing doctoral research in Systems Engineering.

The project is intended for research, experimentation, reproducibility, education, and continued development of climate-aware machine-learning approaches for wireless signal prediction and resilient digital infrastructure.

ClimateNetAI is not a production telecommunications network-management system, and its predictions should be interpreted alongside validation results, dataset limitations, and documented research assumptions.
---

## Purpose

ClimateNetAI investigates how atmospheric and temporal conditions relate to wireless signal behaviour and provides model-based RSSI predictions.

The project supports research into:

- climate-aware wireless communication;
- climate-resilient digital connectivity;
- machine-learning-assisted network analysis;
- model robustness and generalization under environmental variability;
- transparent evaluation of predictive models; and
- responsible and trustworthy application of machine learning to digital infrastructure.

---

## Research Context

ClimateNetAI is informed by ongoing doctoral research undertaken by **Olohimai Juliet Michael** in Systems Engineering at the **African University of Science and Technology (AUST), Abuja, Nigeria**.

The broader research investigates machine-learning-based prediction of wireless signal behaviour under atmospheric variability.

ClimateNetAI translates aspects of this research into an interactive research-software environment through which environmental conditions, trained machine-learning models, validation results, and RSSI predictions can be explored.

---

## Input Variables

ClimateNetAI currently supports the following prediction inputs:

- Temperature
- Atmospheric Pressure
- Relative Humidity
- Month

Depending on the trained monthly model, either three or four features may be used internally.

---

## Target Variable

The prediction target is:

**RSSI — Received Signal Strength Indicator (dBm)**

---

## Machine-Learning Models

ClimateNetAI v1.0.3 supports:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

Model availability may vary by month where validation or reliability checks indicate that a model should not be exposed for prediction.

---

## Application Features

ClimateNetAI v1.0.3 includes:

- monthly RSSI prediction;
- signal-quality classification;
- practical prediction interpretation;
- MAE, RMSE, and R² validation metrics;
- model reliability notices;
- monthly best-model recommendation;
- side-by-side comparison of monthly models;
- monthly R² performance-trend visualization;
- research and data-quality notes;
- downloadable prediction reports; and
- month-aware model availability controls.

---

## Model Validation

ClimateNetAI reports:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination (R²)**

Model recommendations are based primarily on validation R² while MAE and RMSE are also displayed to provide additional context.

A negative validation R² does not mean that the application failed to generate a prediction. It indicates poor out-of-sample generalization relative to predicting the validation-set mean.

The application therefore distinguishes between the ability of a model to produce a numerical prediction and evidence that the model generalizes reliably.

---

## Research and Data-Quality Notes

### June and July

The current modelling dataset contains identical Temperature, Pressure, Relative Humidity, and RSSI observations for June and July.

These months should therefore **not be treated as independent monthly evidence** until the original source data has been verified.

### September

September contains 8 observations in the current modelling dataset.

Its MAE, RMSE, and R² values were reconstructed using **Leave-One-Out Cross Validation (LOOCV)** with:

- Temperature;
- Pressure;
- Relative Humidity; and
- Month Number.

These limitations are retained explicitly in ClimateNetAI to support transparent interpretation of the research results.

---

## Repository Structure

The core repository is organized approximately as follows:

```text
ClimateNetAI/
│
├── app.py
├── modeling_dataset_v1.csv
├── monthly_model_results.csv
├── best_model_per_month_validated.csv
├── requirements.txt
│
├── models/
│   └── monthly_models/
│
├── docs/
│   └── OWNERSHIP_AND_GOVERNANCE.md
│
├── AUTHORS.md
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
└── README.md

## Reproducing the Results

ClimateNetAI includes a reproducibility pipeline for validating the dataset,
preparing the monthly modelling data, training reconstructed machine-learning
models, comparing reproduced results with the archived historical results,
and selecting the best-performing model for each month.

The reproducibility workflow is implemented in:

- `src/data_validation.py`
- `src/preprocessing.py`
- `src/train_models.py`
- `src/evaluate_models.py`
- `src/select_best_models.py`

### 1. Install dependencies

```bash
pip install -r requirements.txt
