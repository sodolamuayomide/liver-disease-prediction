import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Liver Prediction System | Fola",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero {
    background: linear-gradient(135deg, #1a3c5e 0%, #0d6e6e 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
    color: white;
}
.hero p { font-size: 1rem; opacity: 0.85; margin: 0; line-height: 1.6; color: white; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: white;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0d6e6e;
    margin: 1.8rem 0 0.8rem 0;
}
.result-box-sick {
    background-color: #fff0f0;
    border-left: 6px solid #e53e3e;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.result-box-sick h2 { color: #c53030; font-family: 'DM Serif Display', serif; font-size: 1.5rem; margin: 0 0 0.3rem 0; }
.result-box-sick p { color: #2d2d2d; margin: 0; font-size: 0.95rem; }
.result-box-healthy {
    background-color: #f0fff4;
    border-left: 6px solid #38a169;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.result-box-healthy h2 { color: #276749; font-family: 'DM Serif Display', serif; font-size: 1.5rem; margin: 0 0 0.3rem 0; }
.result-box-healthy p { color: #2d2d2d; margin: 0; font-size: 0.95rem; }
.clinical-note {
    background: #f0f7ff;
    border-left: 4px solid #3182ce;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #1a365d;
    margin-top: 1rem;
    line-height: 1.6;
}
.disclaimer {
    background: #fffbeb;
    border: 1px solid #f6e05e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #744210;
    margin-top: 1.5rem;
    line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3c5e, #0d6e6e);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2.5rem; font-size: 1rem;
    font-weight: 600; font-family: 'DM Sans', sans-serif;
    cursor: pointer; width: 100%; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
def load_artifacts():
    with open('liver_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

FEATURE_NAMES = ['age', 'tot_bilirubin', 'direct_bilirubin',
                 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'alkphos']

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🏥 Medical Screening Tool</div>
    <h1>Liver Disease<br>Prediction System</h1>
    <p>Built by <strong>Fola</strong> · This tool uses a machine learning model trained on
    583 patient records from North East Andhra Pradesh, India. It screens for
    potential liver disease based on standard blood test markers. Enter patient
    details below to receive an instant risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ── Patient info ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Patient Information</div>', unsafe_allow_html=True)
patient_name = st.text_input('Patient Name', placeholder='e.g. John Adeyemi')
age = st.number_input('Age', min_value=1, max_value=90, value=45)

# ── Blood test inputs ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🩸 Blood Test Results</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    tot_bilirubin    = st.number_input('Total Bilirubin',    min_value=0.4,  max_value=75.0,   value=1.0,   step=0.1)
    tot_proteins     = st.number_input('Total Proteins',     min_value=63,   max_value=2110,   value=200)
    ag_ratio         = st.number_input('AG Ratio',           min_value=10,   max_value=4929,   value=42)
    alkphos          = st.number_input('Alkaline Phosphatase',min_value=0.3, max_value=2.8,    value=0.9,   step=0.1)
with col2:
    direct_bilirubin = st.number_input('Direct Bilirubin',  min_value=0.1,  max_value=19.7,   value=0.5,   step=0.1)
    albumin          = st.number_input('Albumin',            min_value=10,   max_value=2000,   value=35)
    sgpt             = st.number_input('SGPT',               min_value=2.7,  max_value=9.6,    value=6.5,   step=0.1)
# ── Predict ───────────────────────────────────────────────────────────────────
if st.button('Run Prediction'):
    name_display = patient_name.strip() if patient_name.strip() else "Patient"

    # Apply log transformation to skewed features
    input_df = pd.DataFrame([[
        age,
        np.log1p(tot_bilirubin),
        np.log1p(direct_bilirubin),
        np.log1p(tot_proteins),
        np.log1p(albumin),
        np.log1p(ag_ratio),
        sgpt,
        alkphos
    ]], columns=FEATURE_NAMES)

    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[:, 1][0]
    prediction   = 1 if probability >= 0.45 else 0

    st.markdown("---")
    st.markdown(f"### Results for **{name_display}**")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box-sick">
            <h2>⚠️ HIGH RISK — Liver Disease Detected</h2>
            <p style="color:#555; margin-top:0.3rem;">Liver Disease Probability: <strong>{probability:.1%}</strong></p>
            <p style="margin-top:0.8rem; color:#2d2d2d;">
                Based on the blood test values entered, this patient's overall 
                pattern of markers is consistent with liver disease. Please refer 
                for further clinical evaluation immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-healthy">
            <h2>✅ LOW RISK — No Liver Disease Detected</h2>
            <p style="color:#555; margin-top:0.3rem;">Liver Disease Probability: <strong>{probability:.1%}</strong></p>
            <p style="margin-top:0.8rem; color:#2d2d2d;">
                Based on the blood test values entered, this patient's overall 
                pattern of markers does not strongly indicate liver disease. 
                Routine monitoring is advised.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Why Did The Model Make This Prediction?")
    st.markdown("""
    The chart below shows which features pushed the prediction toward 
    **Sick** (positive values → right) or **Healthy** (negative values → left) 
    for this specific patient.
    """)

    explainer   = shap.LinearExplainer(model, input_scaled)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.values[0],
            feature_names=FEATURE_NAMES
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    # Clinical note
    st.markdown("""
    <div class="clinical-note">
        🔬 <strong>Clinical Context:</strong><br><br>
        <b>Direct Bilirubin</b> — The strongest predictor. Elevated levels signal 
        the liver's failure to conjugate and excrete waste into bile — 
        a direct marker of hepatobiliary dysfunction.<br><br>
        <b>Albumin</b> — Made exclusively by the liver. Falling albumin reflects 
        impaired synthetic function, often an early sign of chronic liver 
        deterioration or acute failure.<br><br>
        <b>AG Ratio</b> — Captures both albumin decline and globulin rise from 
        immune activation. A falling ratio below 1.0 signals advanced liver 
        disease and correlates with poor prognosis.
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for screening purposes only and does
        not constitute a medical diagnosis. Predictions are based on a dataset of 583 patients
        and may not generalise to all populations. SHAP explanations reflect patterns learned 
        from this dataset — not universal clinical truth. Always consult a qualified medical
        professional for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)
