# 🔬 Diabetes Hospital Readmission Prediction Dashboard

An interactive **Streamlit dashboard** that explores and models hospital readmission risk for patients with diabetes using the **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning Repository.

This project demonstrates the **end-to-end data science pipeline**, including data cleaning, exploratory data analysis (EDA), feature engineering, modeling, and evaluation in a healthcare context.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](PASTE_YOUR_STREAMLIT_APP_LINK_HERE)

---

## 📊 Project Overview

Hospital readmissions are costly and often preventable.  
The goal of this project is to:

- Analyze patterns associated with 30-day hospital readmissions
- Build predictive models to identify high-risk patients
- Explore tradeoffs between recall, precision, and clinical impact
- Present insights through an interactive Streamlit dashboard

⚠️ **Disclaimer:** This project is for educational and analytical purposes only and is **not intended for clinical decision-making**.

---

## 🗂 Dataset

This project uses the **Diabetes 130-US hospitals for years 1999–2008** dataset.

- Source: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- Records: ~100,000 hospital encounters
- The dataset is fully **de-identified** and publicly available for research and educational use.

**Citation:**

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014).  
*Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.*  
BioMed Research International, 2014.

---

## 🧠 Modeling Approach

The project includes:
- Data preprocessing and cleaning
- Handling missing values and categorical encoding
- Baseline and advanced machine learning models
- Class imbalance strategies (e.g., undersampling)
- Threshold tuning for healthcare-appropriate tradeoffs
- Evaluation using precision, recall, F1-score, and ROC-AUC

---

## 🚀 How to Run Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py


   
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)