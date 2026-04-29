# Dataset Description

This project uses the **Diabetes 130-US Hospitals dataset** from the UCI Machine Learning Repository.

## Source
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

## Description
The dataset contains over 100,000 hospital encounters for patients with diabetes across 130 U.S. hospitals between 1999 and 2008.

## Files

- `diabetes_deployment.csv`  
  Cleaned dataset used for modeling and deployment.

## Key Features

- Demographics: age, race, gender  
- Utilization: number_inpatient, number_outpatient, number_emergency  
- Clinical: diagnoses (diag_1, diag_2, diag_3), medications  
- Administrative: admission type, discharge disposition  

## Target Variable

- `readmitted` (binary)
  - 0 → Not readmitted  
  - 1 → Readmitted  

## Notes

- Original dataset had 3 classes (<30, >30, No readmission)
- Remapped to binary classification for modeling
- Missing values were handled during preprocessing
