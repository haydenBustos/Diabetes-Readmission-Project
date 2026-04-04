# 🔬 Diabetes Hospital Readmission Prediction Dashboard

An interactive **Streamlit dashboard** that explores a model that classifies hospital readmission risk for patients with diabetes using the **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning Repository.

This project demonstrates the full pipeline, including data cleaning, exploratory data analysis (EDA), feature engineering, modeling, and evaluation in a healthcare context.

[![Open Dashboard in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetes-readmission-dashboard-liwhqp5zv2.streamlit.app/)

---

## 📊 Project Overview

Hospital readmissions are costly and often preventable.  
The goal of this project is to:

- Analyze patterns associated with 30-day hospital readmissions
- Build predictive models to identify high-risk patients
- Present insights through an interactive Streamlit dashboard

⚠️ **Disclaimer:** This project is for educational and analytical purposes only and is **not intended for clinical decision-making**.

---

## 🗂 Dataset

This project uses the **Diabetes 130-US hospitals for years 1999–2008** dataset.

- Source: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- Records: ~100,000 hospital encounters
- The dataset is fully **de-identified** and publicly available for research and educational use.

---

## 🧠 Modeling Approach

The project includes:
- Data preprocessing and cleaning
- Handling missing values and categorical encoding
- Baseline and advanced machine learning models
- Class imbalance strategies (e.g., undersampling)
- Threshold tuning for healthcare-appropriate tradeoffs
- Evaluation using precision, recall, and F1-score

---

## 🚀 How to Run Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
