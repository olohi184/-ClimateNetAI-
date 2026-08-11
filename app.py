from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="ClimateNetAI",
    page_icon="📡",
    layout="wide"
)


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "monthly_models"


# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            background: rgba(245, 247, 250, 0.62);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }

        .snapshot-card, .condition-card {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            background: rgba(250, 250, 250, 0.6);
            min-height: 118px;
        }

        .small-label {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }

        .big-value {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.1;
        }

        .prediction-panel {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            background: rgba(250, 250, 250, 0.55);
            margin-top: 0.5rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }

        .footer-note {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTH_NUMBERS = {
    month: index
    for index, month in enumerate(MONTHS, start=1)
}

MODEL_NAMES = {
    "Random Forest": "Random_Forest",
    "XGBoost": "XGBoost",
    "Decision Tree": "Decision_Tree",
    "Linear Regression": "Linear_Regression"
}


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Prediction Settings")

month = st.sidebar.selectbox(
    "Select Month",
    MONTHS
)

restricted_linear_months = [
    "June",
    "July",
    "August",
    "October",
    "November",
    "December"
]

if month in restricted_linear_months:
    available_models = [
        model_name
        for model_name in MODEL_NAMES.keys()
        if model_name != "Linear Regression"
    ]
else:
    available_models = list(MODEL_NAMES.keys())

model_type = st.sidebar.selectbox(
    "Select Model",
    available_models
)

st.sidebar.divider()
st.sidebar.subheader("Environmental Inputs")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0,
    step=0.1
)

pressure = st.sidebar.number_input(
    "Pressure (hPa)",
    min_value=800.0,
    max_value=1200.0,
    value=1000.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=0.1
)

month_number = MONTH_NUMBERS[month]

predict_button = st.sidebar.button(
    "🔮 Predict RSSI",
    type="primary",
    width="stretch"
)


# ------------------------------------------------------------
# MODEL FILE
# ------------------------------------------------------------

model_filename = f"{month}_{MODEL_NAMES[model_type]}.pkl"
model_path = MODEL_DIR / model_filename


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📡 ClimateNetAI</div>
        <div class="hero-subtitle">
            Climate-Aware Machine Learning for Wireless Signal Prediction
        </div>
        <div style="margin-top: 0.8rem; font-size: 0.92rem; color: #6b7280;">
            Research Prototype • Version 1.0
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Developed by Olohimai Juliet Michael
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# RESEARCH SNAPSHOT
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🔬 Research Snapshot</div>',
    unsafe_allow_html=True
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Dataset</div>
            <div class="big-value">221 records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Environmental Variables</div>
            <div class="big-value">3</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">ML Models</div>
            <div class="big-value">4</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s4:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Prediction Target</div>
            <div class="big-value">RSSI</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CURRENT CONDITIONS
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🌍 Current Prediction Conditions</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Temperature</div>
            <div class="big-value">{temperature:.1f} °C</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Pressure</div>
            <div class="big-value">{pressure:.1f} hPa</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Humidity</div>
            <div class="big-value">{humidity:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Month</div>
            <div class="big-value">{month}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# LIVE PREDICTION
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🔮 Live RSSI Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter environmental conditions in the sidebar and generate a prediction "
    "using one of the trained machine-learning models."
)

if not model_path.exists():
    st.warning(
        f"Model not found for {month} using {model_type}. "
        f"Expected file: {model_filename}"
    )
else:
    st.info(
        f"Ready to predict with **{model_type}** for **{month}**."
    )

if predict_button:

    if not model_path.exists():
        st.error(
            "Prediction cannot be generated because the selected model file is missing."
        )

    else:
        try:
            model = joblib.load(model_path)

            full_input = pd.DataFrame({
                "Temperature": [temperature],
                "Pressure": [pressure],
                "Relative_Humidity": [humidity],
                "Month_Number": [month_number]
            })

            # ------------------------------------------------
            # MODEL FEATURE COMPATIBILITY
            # ------------------------------------------------

            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)

                missing_features = [
                    feature
                    for feature in expected_features
                    if feature not in full_input.columns
                ]

                if missing_features:
                    raise ValueError(
                        f"Model requires unavailable features: {missing_features}"
                    )

                model_input = full_input[expected_features]

            elif hasattr(model, "n_features_in_"):
                feature_count = int(model.n_features_in_)

                if feature_count == 4:
                    expected_features = [
                        "Temperature",
                        "Pressure",
                        "Relative_Humidity",
                        "Month_Number"
                    ]

                elif feature_count == 3:
                    expected_features = [
                        "Temperature",
                        "Pressure",
                        "Relative_Humidity"
                    ]

                else:
                    raise ValueError(
                        f"Unsupported model feature count: {feature_count}"
                    )

                model_input = full_input[expected_features]

            else:
                raise ValueError(
                    "Unable to determine the model's expected input features."
                )

            prediction = float(model.predict(model_input)[0])

            if prediction >= -85:
                quality = "Excellent"
            elif prediction >= -95:
                quality = "Good"
            elif prediction >= -105:
                quality = "Fair"
            else:
                quality = "Poor"

            st.success("Prediction completed successfully.")

            p1, p2 = st.columns([1, 1])

            with p1:
                st.metric(
                    "Predicted RSSI",
                    f"{prediction:.2f} dBm"
                )

            with p2:
                st.metric(
                    "Signal Quality",
                    quality
                )

            # ------------------------------------------------
            # PREDICTION INTERPRETATION
            # ------------------------------------------------

            st.subheader("🧭 Prediction Interpretation")

            if quality == "Excellent":
                interpretation = (
                    "The predicted RSSI indicates a strong signal condition. "
                    "Under these environmental inputs, the connection would generally "
                    "be expected to support stable wireless communication with a strong "
                    "received signal level."
                )

            elif quality == "Good":
                interpretation = (
                    "The predicted RSSI indicates a generally good signal condition. "
                    "Wireless connectivity should remain usable and relatively stable, "
                    "although performance may still vary with network load, interference, "
                    "location, and environmental conditions."
                )

            elif quality == "Fair":
                interpretation = (
                    "The predicted RSSI indicates a moderate signal condition. "
                    "Connectivity may remain usable, but reduced signal margin can make "
                    "performance more sensitive to atmospheric variability, interference, "
                    "mobility, and other propagation effects."
                )

            else:
                interpretation = (
                    "The predicted RSSI indicates a weak signal condition. "
                    "Connectivity may be degraded or unstable, and the link may be more "
                    "susceptible to environmental effects, interference, and coverage limitations."
                )

            st.info(interpretation)

            st.subheader("📊 Signal Assessment")

            a1, a2 = st.columns(2)

            with a1:
                st.info(
                    f"**Signal Quality:** {quality}"
                )

            with a2:
                st.info(
                    f"**Model:** {model_type}"
                )

            # ------------------------------------------------
            # MODEL RELIABILITY NOTICE
            # ------------------------------------------------

            if model_type == "Linear Regression":

                if month == "January":
                    st.success(
                        "✅ **Model Reliability:** January Linear Regression "
                        "was rebuilt, validated, and successfully deployed. "
                        "Its replacement model showed excellent validation performance."
                    )

                elif month == "May":
                    st.warning(
                        "⚠️ **Model Reliability Notice:** The May Linear Regression "
                        "model produces plausible predictions, but its out-of-sample "
                        "validation performance was modest. Interpret this monthly "
                        "prediction with caution."
                    )

                elif month in ["June", "July"]:
                    st.warning(
                        "⚠️ **Data Quality Notice:** June and July contain identical "
                        "Temperature, Pressure, Relative Humidity, and RSSI observations "
                        "in the current modeling dataset. Predictions for these months "
                        "should not be treated as independent monthly evidence until "
                        "the original source data is verified."
                    )

                elif month in ["August", "October", "November", "December"]:
                    st.warning(
                        "⚠️ **Model Reliability Notice:** This monthly Linear Regression "
                        "model showed limited out-of-sample predictive performance during "
                        "validation. The RSSI prediction may be plausible, but it should "
                        "be interpreted cautiously and alongside the model validation results."
                    )

            # ------------------------------------------------
            # MODEL VALIDATION PERFORMANCE
            # ------------------------------------------------

            performance_file = BASE_DIR / "monthly_model_results.csv"

            if performance_file.exists():
                performance_df = pd.read_csv(performance_file)

                performance_df["Month"] = performance_df["Month"].astype(str).str.strip()
                performance_df["Model"] = performance_df["Model"].astype(str).str.strip()

                selected_performance = performance_df[
                    (performance_df["Month"] == month)
                    & (performance_df["Model"] == model_type)
                ]

                if not selected_performance.empty:
                    performance_row = selected_performance.iloc[0]

                    mae_value = float(performance_row["MAE"])
                    rmse_value = float(performance_row["RMSE"])
                    r2_value = float(performance_row["R2"])

                    st.subheader("📈 Model Validation Performance")

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric("MAE", f"{mae_value:.4f}")

                    with m2:
                        st.metric("RMSE", f"{rmse_value:.4f}")

                    with m3:
                        st.metric("R²", f"{r2_value:.4f}")

                    if r2_value >= 0.75:
                        performance_label = "Strong validation performance"
                    elif r2_value >= 0.50:
                        performance_label = "Moderate validation performance"
                    elif r2_value >= 0:
                        performance_label = "Limited validation performance"
                    else:
                        performance_label = "Poor out-of-sample generalization"

                    st.caption(
                        f"Validation assessment: {performance_label}"
                    )

                else:
                    st.info(
                        "Validation metrics are not available "
                        "for this month/model combination."
                    )

            # ------------------------------------------------
            # BEST MODEL FOR SELECTED MONTH
            # ------------------------------------------------

            best_model_file = BASE_DIR / "best_model_per_month_validated.csv"

            if best_model_file.exists():
                best_model_df = pd.read_csv(best_model_file)

                best_model_df["Month"] = best_model_df["Month"].astype(str).str.strip()
                best_model_df["Model"] = best_model_df["Model"].astype(str).str.strip()

                month_best = best_model_df[
                    best_model_df["Month"] == month
                ]

                if not month_best.empty:
                    best_row = month_best.iloc[0]

                    best_model_name = str(best_row["Model"])
                    best_mae = float(best_row["MAE"])
                    best_rmse = float(best_row["RMSE"])
                    best_r2 = float(best_row["R2"])
                    recommendation_status = str(best_row["Recommendation_Status"])

                    st.subheader("🏆 Best Model for Selected Month")

                    if recommendation_status == "Recommended":
                        st.success(
                            f"**Recommended model for {month}: {best_model_name}**  \n"
                            f"R² = {best_r2:.4f} | RMSE = {best_rmse:.4f} | MAE = {best_mae:.4f}"
                        )

                    else:
                        st.warning(
                            f"**Highest-ranked model for {month}: {best_model_name}**  \n"
                            f"R² = {best_r2:.4f} | RMSE = {best_rmse:.4f} | MAE = {best_mae:.4f}  \n"
                            "No model for this month demonstrated positive out-of-sample R²."
                        )

            # ------------------------------------------------
            # MODEL COMPARISON FOR SELECTED MONTH
            # ------------------------------------------------

            st.subheader("📊 Model Comparison for Selected Month")

            month_comparison = performance_df[
                performance_df["Month"] == month
            ][
                ["Model", "MAE", "RMSE", "R2"]
            ].copy()

            if not month_comparison.empty:

                month_comparison = month_comparison.sort_values(
                    by="R2",
                    ascending=False
                )

                month_comparison["MAE"] = month_comparison["MAE"].round(4)
                month_comparison["RMSE"] = month_comparison["RMSE"].round(4)
                month_comparison["R2"] = month_comparison["R2"].round(4)

                month_comparison = month_comparison.rename(
                    columns={
                        "R2": "R²"
                    }
                )

                st.dataframe(
                    month_comparison,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Models are ordered from highest to lowest validation R². "
                    "MAE and RMSE are also shown for comparison."
                )

            else:
                st.info(
                    "Model comparison results are not available "
                    "for the selected month."
                )

            st.subheader("🌍 Environmental Conditions Used")

            result_table = pd.DataFrame({
                "Parameter": [
                    "Temperature",
                    "Pressure",
                    "Relative Humidity",
                    "Month"
                ],
                "Value": [
                    f"{temperature:.2f} °C",
                    f"{pressure:.2f} hPa",
                    f"{humidity:.2f} %",
                    month
                ]
            })

            st.dataframe(
                result_table,
                width="stretch",
                hide_index=True
            )

            with st.expander("Technical model details"):
                st.write(
                    f"**Model file:** `{model_filename}`"
                )
                st.write(
                    f"**Features used:** {', '.join(expected_features)}"
                )
                st.write(
                    f"**Feature count:** {len(expected_features)}"
                )

            report_df = pd.DataFrame({
                "Month": [month],
                "Model": [model_type],
                "Temperature_C": [temperature],
                "Pressure_hPa": [pressure],
                "Relative_Humidity_pct": [humidity],
                "Month_Number": [month_number],
                "Predicted_RSSI_dBm": [prediction],
                "Signal_Quality": [quality],
                "Features_Used": [", ".join(expected_features)],
                "Validation_MAE": [mae_value],
                "Validation_RMSE": [rmse_value],
                "Validation_R2": [r2_value],
                "Validation_Assessment": [performance_label],
                "Monthly_Best_Model": [best_model_name],
                "Monthly_Best_MAE": [best_mae],
                "Monthly_Best_RMSE": [best_rmse],
                "Monthly_Best_R2": [best_r2],
                "Recommendation_Status": [recommendation_status]
            })

            st.download_button(
                "⬇️ Download Prediction Report",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="climatenetai_prediction.csv",
                mime="text/csv",
                width="stretch"
            )

        except Exception as error:
            st.error(
                "An error occurred while generating the prediction."
            )
            st.exception(error)


# ------------------------------------------------------------
# MODEL AVAILABILITY
# ------------------------------------------------------------

st.divider()

with st.expander("🗂️ View Monthly Model Availability"):
    availability_rows = []

    for current_month in MONTHS:
        row = {"Month": current_month}

        for display_name, file_name in MODEL_NAMES.items():
            path = MODEL_DIR / f"{current_month}_{file_name}.pkl"
            row[display_name] = (
                "✅ Available"
                if path.exists()
                else "⚠️ Missing"
            )

        availability_rows.append(row)

    availability_df = pd.DataFrame(availability_rows)

    st.dataframe(
        availability_df,
        width="stretch",
        hide_index=True
    )


# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------

st.divider()
# ------------------------------------------------------------
# MONTHLY MODEL PERFORMANCE TRENDS
# ------------------------------------------------------------

st.subheader("📈 Monthly Model Performance Trends")

trend_file = BASE_DIR / "monthly_model_results.csv"

if trend_file.exists():
    trend_df = pd.read_csv(trend_file)

    trend_df["Month"] = trend_df["Month"].astype(str).str.strip()
    trend_df["Model"] = trend_df["Model"].astype(str).str.strip()

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    trend_df["Month"] = pd.Categorical(
        trend_df["Month"],
        categories=month_order,
        ordered=True
    )

    r2_chart = trend_df.pivot(
        index="Month",
        columns="Model",
        values="R2"
    ).reindex(month_order)

    st.line_chart(
        r2_chart,
        use_container_width=True
    )

    st.caption(
        "Monthly out-of-sample R² performance for the four machine-learning models. "
        "Higher values indicate stronger predictive performance; negative R² values "
        "indicate poor generalization relative to predicting the test-set mean."
    )

    st.info(
        "June and July should be interpreted cautiously because the current modeling "
        "dataset contains identical environmental and RSSI observations for both months."
    )

else:
    st.info(
        "Monthly performance trend data are currently unavailable."
    )

st.divider()

# ------------------------------------------------------------
# RESEARCH AND DATA QUALITY
# ------------------------------------------------------------

st.subheader("🔬 Research & Data Quality")

with st.expander(
    "View validation and data-quality notes",
    expanded=False
):

    st.markdown("**Model validation**")
    st.write(
        "ClimateNetAI reports MAE, RMSE, and R² to provide transparent "
        "information about the predictive performance of each monthly model. "
        "Model recommendations are based primarily on validation R² and should "
        "be interpreted together with the reported error metrics."
    )

    st.markdown("**June and July data-quality notice**")
    st.warning(
        "The current modeling dataset contains identical Temperature, Pressure, "
        "Relative Humidity, and RSSI observations for June and July. Results for "
        "these two months should therefore not be treated as independent monthly "
        "evidence until the original source data has been verified."
    )

    st.markdown("**September validation note**")
    st.info(
        "September contains 8 observations in the current modeling dataset. "
        "Its MAE, RMSE, and R² values were reconstructed using Leave-One-Out "
        "Cross Validation (LOOCV) with Temperature, Pressure, Relative Humidity, "
        "and Month Number as model features."
    )

    st.markdown("**Interpretation of negative R²**")
    st.write(
        "A negative validation R² does not mean that the application failed to "
        "generate a prediction. It indicates that the model generalized poorly "
        "on the validation observations relative to a simple mean-prediction "
        "baseline. ClimateNetAI therefore avoids presenting such models as "
        "recommended models."
    )

    st.caption(
        "These notes are included to support transparent interpretation of "
        "ClimateNetAI outputs and should be considered when using results for "
        "research, reporting, or decision support."
    )

st.divider()

st.subheader("ℹ️ About ClimateNetAI")

st.write(
    """
ClimateNetAI is a climate-aware machine-learning research application
developed to demonstrate RSSI prediction from real-world atmospheric
measurements.

The deployment supports both legacy 3-feature monthly models
(Temperature, Pressure and Relative Humidity) and newer 4-feature models
that additionally use Month Number. The saved trained models are not
altered by the dashboard.
"""
)

st.markdown(
    '<div class="footer-note">Research demonstration — interpret predictions together with model-validation results and study limitations.</div>',
    unsafe_allow_html=True
)
