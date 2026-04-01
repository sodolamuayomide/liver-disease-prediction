# 🏥 Liver Disease Prediction System

**Built by:** Fola  
**Dataset:** Indian Liver Patient Dataset (ILPD)  
**Tool:** Python, Scikit-learn, Streamlit  

---

## Problem Statement
Liver disease is a silent killer. Many patients show no symptoms until 
the disease has progressed significantly. Early screening using blood 
test markers can save lives.

This project builds a machine learning classification model that predicts 
whether a patient has liver disease based on 8 blood test measurements.

---

## Key Engineering Decision
In medical classification, not all errors are equal.

- **False Negative** → Model says healthy, patient is sick → patient goes home untreated → dangerous
- **False Positive** → Model says sick, patient is healthy → extra tests, temporary anxiety → acceptable

Therefore, I optimized for **Recall** over accuracy throughout this project.
A model that catches 93% of sick patients is worth more than one with 
higher accuracy but more missed diagnoses.

---

## Dataset
- 583 patient records from North East Andhra Pradesh, India
- 416 liver patients (71%), 167 healthy (29%)
- 10 features including bilirubin levels, liver enzymes, and protein markers
- Source: UCI Machine Learning Repository

---

## Project Pipeline
1. **Data Cleaning** — Fixed target encoding, filled missing values, encoded gender
2. **EDA** — Identified class imbalance and strong predictive features
3. **Feature Selection** — Removed weak features (sgot, gender) based on importance analysis
4. **Log Transformation** — Fixed skewed distributions in 5 features
5. **Modeling** — Tested Logistic Regression, Random Forest, XGBoost
6. **Threshold Tuning** — Lowered decision threshold from 0.5 → 0.45 to reduce False Negatives
7. **Deployment** — Built Streamlit web app for real-time predictions

---

## Results

| Model | Accuracy | Recall (Sick) | False Negatives |
|---|---|---|---|
| Logistic Regression (baseline) | 0.76 | 0.92 | 7 |
| Random Forest | 0.74 | 0.86 | 12 |
| XGBoost | 0.72 | 0.83 | 15 |
| **Final Model (tuned LR)** | **0.73** | **0.93** | **6** |

Logistic Regression outperformed more complex models — demonstrating that 
algorithm complexity does not guarantee better results, especially on small datasets.

---

## How To Run

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost streamlit
```

**Run the app:**
```bash
streamlit run app.py
```

---

## Project Structure
```
liver_patient_project/
├── liver_patient_analysis.ipynb  # Full analysis notebook
├── app.py                        # Streamlit web application
├── liver_model_v2.pkl            # Trained model
├── scaler_v2.pkl                 # Feature scaler
├── indian_liver_patient.csv      # Dataset
└── README.md                     # This file
```

---

## Key Learnings
- Accuracy is a misleading metric for imbalanced medical datasets
- Simpler models can outperform complex ones on small datasets
- Threshold tuning is a powerful and underused technique
- Feature distributions matter — log transformation improved model consistency


## ⚠️ Limitations
- Small dataset (583 patients) from one geographic region — 
  may not generalise globally
- Dataset uses non-standard units — app inputs must match 
  training data units, not clinical reference ranges
- SHAP analysis revealed albumin's directional influence 
  contradicts clinical literature — likely due to class 
  imbalance and dataset size
- Model is a screening tool only — not a diagnostic instrument


## 🔭 Future Work
- Collect larger, more balanced dataset with standardised units
- Explore temporal modeling — tracking patient markers over 
  time rather than single snapshots
- Integrate ICU data for critical care prediction
- Validate model against clinical expert judgment
- Explore deep learning approaches with larger datasets