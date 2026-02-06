import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="Diabetes Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data(show_spinner="Loading dataset...")
def get_diabetes_data() -> pd.DataFrame:
    """
    Load the diabetes deployment dataset from the repo.
    Expected path (relative to this file):
      Diabetes-Readmission-Project/data/diabetes_deployment.csv
    """
    DATA_FILENAME = Path(__file__).parent / "data" / "diabetes_deployment.csv"
    df = pd.read_csv(DATA_FILENAME)

    # Light cleanup for consistent filtering / display
    for col in ["race", "gender", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Ensure numeric columns are numeric (if they aren't already)
    for col in ["age", "time_in_hospital", "num_medications", "num_lab_procedures"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure readmitted is 0/1
    if "readmitted" in df.columns:
        df["readmitted"] = pd.to_numeric(df["readmitted"], errors="coerce").fillna(0).astype(int)

    return df


df = get_diabetes_data()

# -----------------------------------------------------------------------------
# Helpers

def age_to_band_label(age_midpoint: float) -> str:
    """
    In the UCI dataset, age is often bucketed into 10-year bands.
    Your file appears to use midpoints like 5, 15, 25, ... 95.
    We'll display them as 0–10, 10–20, ..., 90–100.
    """
    if pd.isna(age_midpoint):
        return "Unknown"
    a = int(round(age_midpoint))
    low = max(0, a - 5)
    high = a + 5
    return f"{low:02d}–{high:02d}"


# -----------------------------------------------------------------------------
# Header

st.title("🏥 Diabetes Hospital Readmission Dashboard")
st.caption(
    "Exploratory demographics + readmission rates using the UCI Diabetes 130-US Hospitals dataset (1999–2008). "
    "For educational/portfolio use only — not clinical decision-making."
)

st.divider()

# -----------------------------------------------------------------------------
# Sidebar filters

with st.sidebar:
    st.header("Filters")

    # Basic filters
    races = sorted(df["race"].dropna().unique().tolist()) if "race" in df.columns else []
    genders = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []

    selected_races = st.multiselect("Race", races, default=races[:3] if len(races) >= 3 else races)
    selected_genders = st.multiselect("Gender", genders, default=genders)

    # Age slider
    if "age" in df.columns:
        min_age = int(df["age"].min())
        max_age = int(df["age"].max())
        age_range = st.slider("Age band midpoint (10-year buckets)", min_age, max_age, (min_age, max_age), step=10)
    else:
        age_range = None

    # Readmission filter
    readmission_mode = st.radio(
        "Readmission",
        options=["All", "Readmitted only", "Not readmitted only"],
        index=0
    )

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
# Charts (keep it simple: demographics + readmission rates)

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
# Optional: show a small preview table (useful for debugging deployment)

with st.expander("Preview filtered data (first 50 rows)"):
    st.dataframe(filtered.head(50), use_container_width=True)
