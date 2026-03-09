import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

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
    """
    Load the diabetes deployment dataset from the repo.
    Expected path:
      data/diabetes_deployment.csv
    """
    data_filename = DATA_DIR / "diabetes_deployment.csv"
    df = pd.read_csv(data_filename)

    # Light cleanup for consistent filtering / display
    for col in ["race", "gender", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Ensure numeric columns are numeric
    for col in ["age", "time_in_hospital", "num_medications", "num_lab_procedures"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure readmitted is 0/1
    if "readmitted" in df.columns:
        df["readmitted"] = pd.to_numeric(df["readmitted"], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return joblib.load(ARTIFACTS_DIR / "lightgbm_model.joblib")


@st.cache_resource(show_spinner="Loading model features...")
def load_model_features():
    return joblib.load(ARTIFACTS_DIR / "model_features.joblib")


df = get_diabetes_data()
model = load_model()
model_features = load_model_features()

# -----------------------------------------------------------------------------
# Helpers

def age_to_band_label(age_midpoint: float) -> str:
    """
    In the UCI dataset, age is often bucketed into 10-year bands.
    Midpoints like 5, 15, 25, ... 95 are displayed as 0–10, 10–20, ..., 90–100.
    """
    if pd.isna(age_midpoint):
        return "Unknown"
    a = int(round(age_midpoint))
    low = max(0, a - 5)
    high = a + 5
    return f"{low:02d}–{high:02d}"


def prepare_model_input(input_df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
    Apply the same basic preprocessing pattern used during training:
    - fill string NA values
    - pd.get_dummies()
    - align columns to saved training feature order
    """
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


# -----------------------------------------------------------------------------
# Header

st.title("🏥 Diabetes Hospital Readmission Dashboard")
st.caption(
    "Exploratory demographics + readmission rates using the UCI Diabetes 130-US Hospitals dataset (1999–2008). "
    "For educational/portfolio use only — not clinical decision-making."
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
# Quick metrics

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

# -----------------------------------------------------------------------------
# Charts

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
        st.info("Not enough data after filtering to plot age readmission rates.")

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
        st.info("Not enough data after filtering to plot race readmission rates.")

st.divider()

c3, c4 = st.columns(2)

with c3:
    st.subheader("Readmission rate by gender")
    if total and "gender" in filtered.columns and "readmitted" in filtered.columns:
        gender_rate = (
            filtered.groupby("gender", as_index=False)["readmitted"]
            .mean()
            .sort_values("gender")
        )
        gender_rate["readmit_rate"] = gender_rate["readmitted"] * 100
        st.bar_chart(gender_rate, x="gender", y="readmit_rate")
    else:
        st.info("Not enough data after filtering to plot gender readmission rates.")

with c4:
    st.subheader("Top medical specialties (count)")
    if total and "medical_specialty" in filtered.columns:
        spec_counts = (
            filtered["medical_specialty"]
            .value_counts()
            .head(10)
            .rename_axis("medical_specialty")
            .reset_index(name="count")
        )
        st.bar_chart(spec_counts, x="medical_specialty", y="count")
    else:
        st.info("Not enough data after filtering to show medical specialties.")

st.divider()

# -----------------------------------------------------------------------------
# Prediction section
# Uses a real encounter row from the filtered dataset so the model receives
# the full feature set it was trained on.

st.header("Patient Readmission Prediction")
st.caption(
    "Select an encounter from the filtered dataset, optionally adjust a few fields, "
    "and generate a readmission prediction with the trained LightGBM model."
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
                editable_row.at[editable_row.index[0], "age"] = st.selectbox(
                    "Age",
                    options=sorted(df["age"].dropna().unique().tolist()),
                    index=sorted(df["age"].dropna().unique().tolist()).index(editable_row.iloc[0]["age"])
                    if editable_row.iloc[0]["age"] in sorted(df["age"].dropna().unique().tolist())
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

        result_left, result_mid = st.columns(2)
        result_left.metric("Predicted readmission probability", f"{pred_proba:.1%}")
        result_mid.metric("Predicted class", "Readmitted" if pred_class == 1 else "Not readmitted")

        st.caption(
            "This prediction is generated from the trained LightGBM model using the selected encounter as a base profile. "
            "It is for demonstration only and not for clinical use."
        )

        with st.expander("Preview model-ready input row"):
            st.dataframe(model_input, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Optional: preview filtered data

with st.expander("Preview filtered data (first 50 rows)"):
    st.dataframe(filtered.head(50), use_container_width=True)