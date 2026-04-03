import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="Diabetes Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Paths

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data(show_spinner="Loading dataset...")
def get_diabetes_data() -> pd.DataFrame:
    data_filename = DATA_DIR / "diabetes_deployment.csv"
    df = pd.read_csv(data_filename)

    for col in ["race", "gender", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    for col in ["age", "time_in_hospital", "num_medications", "num_lab_procedures"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "readmitted" in df.columns:
        df["readmitted"] = pd.to_numeric(df["readmitted"], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return joblib.load(ARTIFACTS_DIR / "lightgbm_model.joblib")


@st.cache_resource(show_spinner="Loading model features...")
def load_model_features():
    return joblib.load(ARTIFACTS_DIR / "model_features.joblib")


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


df = get_diabetes_data()
model = load_model()
model_features = load_model_features()
shap_explainer = load_shap_explainer(model)

# -----------------------------------------------------------------------------
# Subgroup performance metadata

subgroup_performance = {
    "race": {
        "AfricanAmerican": {"recall_1": 0.60, "support": 1572, "accuracy": 0.64},
        "Asian": {"recall_1": 0.67, "support": 62, "accuracy": 0.77},
        "Caucasian": {"recall_1": 0.65, "support": 5547, "accuracy": 0.63},
        "Hispanic": {"recall_1": 0.62, "support": 152, "accuracy": 0.64},
        "Other": {"recall_1": 0.42, "support": 97, "accuracy": 0.67},
    },
    "age": {
        5: {"recall_1": 0.00, "support": 12, "accuracy": 0.67},
        15: {"recall_1": 0.33, "support": 71, "accuracy": 0.65},
        25: {"recall_1": 0.73, "support": 133, "accuracy": 0.77},
        35: {"recall_1": 0.51, "support": 309, "accuracy": 0.67},
        45: {"recall_1": 0.50, "support": 783, "accuracy": 0.64},
        55: {"recall_1": 0.53, "support": 1304, "accuracy": 0.64},
        65: {"recall_1": 0.65, "support": 1683, "accuracy": 0.62},
        75: {"recall_1": 0.68, "support": 1898, "accuracy": 0.62},
        85: {"recall_1": 0.73, "support": 1237, "accuracy": 0.63},
    },
    "payer": {
        "MISSING": {"recall_1": 0.61, "support": 3275, "accuracy": 0.63},
        "BC": {"recall_1": 0.48, "support": 322, "accuracy": 0.66},
        "CH": {"recall_1": 0.25, "support": 14, "accuracy": 0.64},
        "CM": {"recall_1": 0.65, "support": 129, "accuracy": 0.57},
        "CP": {"recall_1": 0.55, "support": 193, "accuracy": 0.65},
        "DM": {"recall_1": 0.58, "support": 43, "accuracy": 0.65},
        "HM": {"recall_1": 0.60, "support": 432, "accuracy": 0.64},
        "MC": {"recall_1": 0.71, "support": 2061, "accuracy": 0.62},
        "MD": {"recall_1": 0.64, "support": 226, "accuracy": 0.62},
        "MP": {"recall_1": 0.80, "support": 6, "accuracy": 0.67},
        "OG": {"recall_1": 0.55, "support": 73, "accuracy": 0.73},
        "OT": {"recall_1": 0.67, "support": 4, "accuracy": 0.50},
        "PO": {"recall_1": 0.39, "support": 52, "accuracy": 0.71},
        "SI": {"recall_1": 0.67, "support": 5, "accuracy": 0.60},
        "SP": {"recall_1": 0.65, "support": 388, "accuracy": 0.63},
        "UN": {"recall_1": 0.51, "support": 191, "accuracy": 0.75},
        "WC": {"recall_1": 0.00, "support": 16, "accuracy": 0.69},
    }
}

# -----------------------------------------------------------------------------
# Helpers

def age_to_band_label(age_midpoint: float) -> str:
    if pd.isna(age_midpoint):
        return "Unknown"
    a = int(round(age_midpoint))
    low = max(0, a - 5)
    high = a + 5
    return f"{low:02d}–{high:02d}"


def prepare_model_input(input_df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    df_copy = input_df.copy()

    for col in ["race", "gender", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna("Unknown").astype(str)

    for col in ["age", "time_in_hospital", "num_medications", "num_lab_procedures"]:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    encoded = pd.get_dummies(df_copy)
    encoded = encoded.reindex(columns=feature_order, fill_value=0)
    return encoded


def normalize_payer_code(payer_value: str) -> str:
    if payer_value in ["Unknown", "UNKNOWN", "nan", "None", ""]:
        return "MISSING"
    return payer_value


def get_reliability_label(recall: float, support: int) -> str:
    if support < 30:
        return "⚠️ Very limited reliability"
    elif support < 100:
        return "⚠️ Lower reliability"
    elif recall >= 0.65:
        return "✅ Higher reliability"
    elif recall >= 0.55:
        return "🟡 Moderate reliability"
    else:
        return "❗ Lower reliability"


def reliability_score(recall: float, support: int) -> float:
    """
    Penalize tiny subgroup sizes so very small samples do not look overly trustworthy.
    Output is a 0-1 score.
    """
    if support < 30:
        penalty = 0.55
    elif support < 100:
        penalty = 0.75
    else:
        penalty = 1.0
    return recall * penalty


def score_to_trust_label(score: float) -> tuple[str, str]:
    if score >= 0.65:
        return "High", "✅"
    elif score >= 0.50:
        return "Medium", "🟡"
    else:
        return "Low", "❗"


def render_subgroup_metric(container, title: str, subgroup_value, subgroup_dict: dict):
    if subgroup_value in subgroup_dict:
        data = subgroup_dict[subgroup_value]
        recall = data["recall_1"]
        support = data["support"]
        accuracy = data["accuracy"]
        label = get_reliability_label(recall, support)

        container.metric(title, f"{recall:.2f}")
        container.caption(label)
        container.caption(f"Support: {support:}")
    else:
        container.metric(title, "n/a")
        container.caption("No subgroup analysis available")


def build_trust_gauge_html(score: float, label: str, icon: str) -> str:
    score_pct = max(0, min(100, int(round(score * 100))))

    if label == "High":
        active_color = "#4CAF50"
    elif label == "Medium":
        active_color = "#F4B400"
    else:
        active_color = "#DB4437"

    return f"""
    <div style="background:#F8FAFC; padding:1rem 1.25rem; border-radius:12px; border:1px solid #E5E7EB;">
        <div style="font-size:1.05rem; font-weight:600; margin-bottom:0.5rem;">
            Overall subgroup trust gauge
        </div>
        <div style="font-size:2rem; font-weight:700; margin-bottom:0.35rem;">
            {icon} {label} Trust
        </div>
        <div style="font-size:0.95rem; color:#374151; margin-bottom:0.75rem;">
            Composite subgroup reliability score: {score_pct} / 100
        </div>
        <div style="display:flex; width:100%; height:16px; border-radius:999px; overflow:hidden; background:#E5E7EB; margin-bottom:0.6rem;">
            <div style="width:33.33%; background:{active_color if label == 'Low' else '#DB4437'};"></div>
            <div style="width:33.33%; background:{active_color if label == 'Medium' else '#F4B400'};"></div>
            <div style="width:33.34%; background:{active_color if label == 'High' else '#4CAF50'};"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#4B5563;">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
        </div>
    </div>
    """


# -----------------------------------------------------------------------------
# Header

st.title("🏥 Diabetes Hospital Readmission Dashboard")
st.caption(
    "Exploratory demographics + readmission rates using the UCI Diabetes 130-US Hospitals dataset (1999–2008)."
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "EDA Dashboard",
    "Prediction",
    "Model Interpretability"
])

# -----------------------------------------------------------------------------
# Sidebar filters

with st.sidebar:
    st.header("Filters")

    races = sorted(df["race"].dropna().unique().tolist()) if "race" in df.columns else []
    genders = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []

    selected_races = st.multiselect("Race", races, default=races[:3] if len(races) >= 3 else races)
    selected_genders = st.multiselect("Gender", genders, default=genders)

    if "age" in df.columns:
        min_age = int(df["age"].min())
        max_age = int(df["age"].max())
        age_range = st.slider(
            "Age band midpoint (10-year buckets)",
            min_age,
            max_age,
            (min_age, max_age),
            step=10,
        )
    else:
        age_range = None

    readmission_mode = st.radio(
        "Readmission",
        options=["All", "Readmitted only", "Not readmitted only"],
        index=0,
    )

# -----------------------------------------------------------------------------
# Apply filters

filtered = df.copy()

if "race" in filtered.columns and selected_races:
    filtered = filtered[filtered["race"].isin(selected_races)]

if "gender" in filtered.columns and selected_genders:
    filtered = filtered[filtered["gender"].isin(selected_genders)]

if age_range and "age" in filtered.columns:
    filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

if "readmitted" in filtered.columns:
    if readmission_mode == "Readmitted only":
        filtered = filtered[filtered["readmitted"] == 1]
    elif readmission_mode == "Not readmitted only":
        filtered = filtered[filtered["readmitted"] == 0]

# -----------------------------------------------------------------------------
# Tab 1: EDA Dashboard

with tab1:
    left, mid, right, far_right = st.columns(4)

    total = len(filtered)
    readmit_rate = (filtered["readmitted"].mean() * 100) if total and "readmitted" in filtered.columns else 0.0
    avg_los = filtered["time_in_hospital"].mean() if total and "time_in_hospital" in filtered.columns else float("nan")
    avg_meds = filtered["num_medications"].mean() if total and "num_medications" in filtered.columns else float("nan")

    left.metric("Encounters (filtered)", f"{total:,}")
    mid.metric("Readmission rate", f"{readmit_rate:.1f}%")
    right.metric("Avg length of stay (days)", f"{avg_los:.2f}" if pd.notna(avg_los) else "n/a")
    far_right.metric("Avg # medications", f"{avg_meds:.2f}" if pd.notna(avg_meds) else "n/a")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Readmission rate by age band")
        if total and "age" in filtered.columns and "readmitted" in filtered.columns:
            tmp = filtered.copy()
            tmp["age_band"] = tmp["age"].apply(age_to_band_label)
            age_rate = (
                tmp.groupby("age_band", as_index=False)["readmitted"]
                .mean()
                .sort_values("age_band")
            )
            age_rate["readmit_rate"] = age_rate["readmitted"] * 100
            st.bar_chart(age_rate, x="age_band", y="readmit_rate")
        else:
            st.info("You have removed all groups. Add groups for visuals.")

    with c2:
        st.subheader("Readmission rate by race")
        if total and "race" in filtered.columns and "readmitted" in filtered.columns:
            race_rate = (
                filtered.groupby("race", as_index=False)["readmitted"]
                .mean()
                .sort_values("readmitted", ascending=False)
            )
            race_rate["readmit_rate"] = race_rate["readmitted"] * 100
            st.bar_chart(race_rate, x="race", y="readmit_rate")
        else:
            st.info("You have removed all groups. Add groups for visuals.")

    st.divider()

    st.subheader("Global feature importance (filtered sample)")
    st.caption(
        "Mean absolute SHAP values from the currently filtered patient sample. "
        "Higher values indicate features that had greater overall influence on model predictions."
    )

    if filtered.empty:
        st.info("No filtered data available to calculate global SHAP importance.")
    else:
        global_source = filtered.drop(columns=["readmitted"], errors="ignore").copy()
        sample_size = min(200, len(global_source))
        global_sample = global_source.sample(sample_size, random_state=42)

        global_model_input = prepare_model_input(global_sample, model_features)
        shap_values_global = shap_explainer.shap_values(global_model_input)

        if isinstance(shap_values_global, list):
            global_vals = shap_values_global[1]
        else:
            global_vals = shap_values_global

        global_importance = pd.DataFrame({
            "feature": global_model_input.columns,
            "mean_abs_shap": abs(global_vals).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).head(15)

        st.bar_chart(global_importance.set_index("feature")["mean_abs_shap"])

        with st.expander("Preview global SHAP importance table"):
            st.dataframe(global_importance, use_container_width=True)

    st.divider()

    with st.expander("Preview filtered data (first 50 rows)"):
        st.dataframe(filtered.head(50), use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Prediction

with tab2:
    st.header("Patient Readmission Prediction")
    st.caption(
        "Select an encounter from the filtered dataset, optionally adjust fields, and"
        "generate a readmission prediction with the trained LightGBM model."
    )

    if filtered.empty:
        st.warning("No rows available after filtering. Adjust the sidebar filters to make predictions.")
    else:
        prediction_source = filtered.reset_index(drop=False).rename(columns={"index": "original_index"})

        selector_cols = ["original_index"]
        for col in ["age", "race", "gender", "medical_specialty", "time_in_hospital"]:
            if col in prediction_source.columns:
                selector_cols.append(col)

        st.subheader("Choose an encounter")
        selected_position = st.selectbox(
            "Encounter row",
            options=prediction_source.index.tolist(),
            format_func=lambda i: (
                f"Row {prediction_source.loc[i, 'original_index']} | "
                f"Age: {prediction_source.loc[i, 'age'] if 'age' in prediction_source.columns else 'n/a'} | "
                f"Race: {prediction_source.loc[i, 'race'] if 'race' in prediction_source.columns else 'n/a'} | "
                f"Gender: {prediction_source.loc[i, 'gender'] if 'gender' in prediction_source.columns else 'n/a'}"
            ),
        )

        base_row = prediction_source.loc[[selected_position]].copy()

        display_cols = [c for c in selector_cols if c in base_row.columns]
        with st.expander("Preview selected encounter"):
            st.dataframe(base_row[display_cols], use_container_width=True)

        st.subheader("Optional adjustments")
        editable_row = base_row.drop(columns=["readmitted"], errors="ignore").copy()

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if "age" in editable_row.columns:
                    age_options = sorted(df["age"].dropna().unique().tolist())
                    editable_row.at[editable_row.index[0], "age"] = st.selectbox(
                        "Age",
                        options=age_options,
                        index=age_options.index(editable_row.iloc[0]["age"])
                        if editable_row.iloc[0]["age"] in age_options
                        else 0,
                    )
                if "race" in editable_row.columns:
                    race_options = sorted(df["race"].dropna().astype(str).unique().tolist())
                    editable_row.at[editable_row.index[0], "race"] = st.selectbox(
                        "Race",
                        options=race_options,
                        index=race_options.index(str(editable_row.iloc[0]["race"]))
                        if str(editable_row.iloc[0]["race"]) in race_options
                        else 0,
                    )
                if "gender" in editable_row.columns:
                    gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist())
                    editable_row.at[editable_row.index[0], "gender"] = st.selectbox(
                        "Gender",
                        options=gender_options,
                        index=gender_options.index(str(editable_row.iloc[0]["gender"]))
                        if str(editable_row.iloc[0]["gender"]) in gender_options
                        else 0,
                    )

            with col2:
                if "time_in_hospital" in editable_row.columns:
                    editable_row.at[editable_row.index[0], "time_in_hospital"] = st.number_input(
                        "Time in hospital",
                        min_value=1,
                        max_value=30,
                        value=int(editable_row.iloc[0]["time_in_hospital"]) if pd.notna(editable_row.iloc[0]["time_in_hospital"]) else 1,
                    )
                if "num_medications" in editable_row.columns:
                    editable_row.at[editable_row.index[0], "num_medications"] = st.number_input(
                        "Number of medications",
                        min_value=0,
                        max_value=100,
                        value=int(editable_row.iloc[0]["num_medications"]) if pd.notna(editable_row.iloc[0]["num_medications"]) else 0,
                    )
                if "num_lab_procedures" in editable_row.columns:
                    editable_row.at[editable_row.index[0], "num_lab_procedures"] = st.number_input(
                        "Number of lab procedures",
                        min_value=0,
                        max_value=150,
                        value=int(editable_row.iloc[0]["num_lab_procedures"]) if pd.notna(editable_row.iloc[0]["num_lab_procedures"]) else 0,
                    )

            with col3:
                if "payer_code" in editable_row.columns:
                    payer_options = sorted(df["payer_code"].fillna("Unknown").astype(str).unique().tolist())
                    editable_row.at[editable_row.index[0], "payer_code"] = st.selectbox(
                        "Payer code",
                        options=payer_options,
                        index=payer_options.index(str(editable_row.iloc[0]["payer_code"]))
                        if str(editable_row.iloc[0]["payer_code"]) in payer_options
                        else 0,
                    )
                if "medical_specialty" in editable_row.columns:
                    specialty_options = sorted(df["medical_specialty"].fillna("Unknown").astype(str).unique().tolist())
                    editable_row.at[editable_row.index[0], "medical_specialty"] = st.selectbox(
                        "Medical specialty",
                        options=specialty_options,
                        index=specialty_options.index(str(editable_row.iloc[0]["medical_specialty"]))
                        if str(editable_row.iloc[0]["medical_specialty"]) in specialty_options
                        else 0,
                    )
                if "diag_1" in editable_row.columns:
                    diag1_options = sorted(df["diag_1"].fillna("Unknown").astype(str).unique().tolist())
                    editable_row.at[editable_row.index[0], "diag_1"] = st.selectbox(
                        "Primary diagnosis",
                        options=diag1_options,
                        index=diag1_options.index(str(editable_row.iloc[0]["diag_1"]))
                        if str(editable_row.iloc[0]["diag_1"]) in diag1_options
                        else 0,
                    )

            submitted = st.form_submit_button("Predict readmission risk")

        if submitted:
            model_input = prepare_model_input(editable_row, model_features)

            pred_class = int(model.predict(model_input)[0])
            pred_proba = float(model.predict_proba(model_input)[0][1])

            st.session_state["latest_model_input"] = model_input.copy()
            st.session_state["latest_pred_class"] = pred_class
            st.session_state["latest_pred_proba"] = pred_proba
            st.session_state["latest_editable_row"] = editable_row.copy()

            result_left, result_mid = st.columns(2)
            result_left.metric("Predicted readmission probability", f"{pred_proba:.1%}")
            result_mid.metric("Predicted class", "Readmitted" if pred_class == 1 else "Not readmitted")

            st.divider()
            st.subheader("Prediction Reliability Across Similar Patient Subgroups")

            patient_race = str(editable_row.iloc[0].get("race", "Unknown"))
            patient_age = editable_row.iloc[0].get("age", None)
            patient_payer = normalize_payer_code(str(editable_row.iloc[0].get("payer_code", "Unknown")))

            if pd.notna(patient_age):
                patient_age = int(patient_age)

            subgroup_scores = []

            if patient_race in subgroup_performance["race"]:
                race_data = subgroup_performance["race"][patient_race]
                subgroup_scores.append(reliability_score(race_data["recall_1"], race_data["support"]))

            if patient_age in subgroup_performance["age"]:
                age_data = subgroup_performance["age"][patient_age]
                subgroup_scores.append(reliability_score(age_data["recall_1"], age_data["support"]))

            if patient_payer in subgroup_performance["payer"]:
                payer_data = subgroup_performance["payer"][patient_payer]
                subgroup_scores.append(reliability_score(payer_data["recall_1"], payer_data["support"]))

            if subgroup_scores:
                overall_trust_score = sum(subgroup_scores) / len(subgroup_scores)
                trust_label, trust_icon = score_to_trust_label(overall_trust_score)
                st.markdown(build_trust_gauge_html(overall_trust_score, trust_label, trust_icon), unsafe_allow_html=True)
            else:
                st.info("No subgroup trust information is available for this patient profile.")

            st.write("")

            rel1, rel2, rel3 = st.columns(3)

            render_subgroup_metric(
                rel1,
                "Race subgroup recall",
                patient_race,
                subgroup_performance["race"]
            )

            render_subgroup_metric(
                rel2,
                "Age subgroup recall",
                patient_age,
                subgroup_performance["age"]
            )

            render_subgroup_metric(
                rel3,
                "Payer subgroup recall",
                patient_payer,
                subgroup_performance["payer"]
            )

# -----------------------------------------------------------------------------
# Tab 3: Model Interpretability

with tab3:
    st.header("Model Interpretability")
    st.caption(
        "SHAP shows which features pushed the prediction higher or lower for the current patient profile. "
    )

    if "latest_model_input" not in st.session_state:
        st.info("Run a prediction in the Prediction tab to generate SHAP plots.")
    else:
        latest_model_input = st.session_state["latest_model_input"]
        latest_pred_class = st.session_state["latest_pred_class"]
        latest_pred_proba = st.session_state["latest_pred_proba"]

        st.subheader("Last prediction summary")
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted readmission probability", f"{latest_pred_proba:.1%}")
        col_b.metric("Predicted class", "Readmitted" if latest_pred_class == 1 else "Not readmitted")

        st.subheader("SHAP Waterfall Plot")
        shap_values_single = shap_explainer.shap_values(latest_model_input)

        if isinstance(shap_values_single, list):
            single_vals = shap_values_single[1][0]
            base_value = shap_explainer.expected_value[1]
        else:
            single_vals = shap_values_single[0]
            base_value = shap_explainer.expected_value

        explanation = shap.Explanation(
            values=single_vals,
            base_values=base_value,
            data=latest_model_input.iloc[0].values,
            feature_names=latest_model_input.columns.tolist()
        )

        plt.close("all")
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig_local = plt.gcf()
        st.pyplot(fig_local, clear_figure=True)

        with st.expander("Preview SHAP values table"):
            local_shap_df = pd.DataFrame({
                "feature": latest_model_input.columns,
                "shap_value": single_vals
            })
            local_shap_df["abs_shap_value"] = local_shap_df["shap_value"].abs()
            local_shap_df = local_shap_df.sort_values("abs_shap_value", ascending=False).head(15)
            st.dataframe(local_shap_df, use_container_width=True)