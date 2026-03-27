import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Liver Prediction System | Fola",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
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
.hero p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0;
    line-height: 1.6;
    color: white;
}
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
.result-box-sick h2 {
    color: #c53030;
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin: 0 0 0.3rem 0;
}
.result-box-sick p {
    color: #2d2d2d;
    margin: 0;
    font-size: 0.95rem;
}
.result-box-healthy {
    background-color: #f0fff4;
    border-left: 6px solid #38a169;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.result-box-healthy h2 {
    color: #276749;
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin: 0 0 0.3rem 0;
}
.result-box-healthy p {
    color: #2d2d2d;
    margin: 0;
    font-size: 0.95rem;
}
.reason-item-red {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #1a1a1a;
}
.reason-item-yellow {
    background: #fffff0;
    border: 1px solid #f6e05e;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #1a1a1a;
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
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    width: 100%;
    margin-top: 1rem;
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

# ── Normal ranges ─────────────────────────────────────────────────────────────
NORMAL_RANGES = {
    'Total Bilirubin':      (0.1, 1.2,  'mg/dL', 'tot_bilirubin'),
    'Direct Bilirubin':     (0.0, 0.3,  'mg/dL', 'direct_bilirubin'),
    'Total Proteins':       (150, 400,  'U/L',   'tot_proteins'),
    'Albumin':              (35,  55,   'g/dL',  'albumin'),
    'AG Ratio':             (1,   2,    '',       'ag_ratio'),
    'SGPT':                 (0,   56,   'U/L',   'sgpt'),
    'Alkaline Phosphotase': (0.3, 1.5,  'U/L',   'alkphos'),
}

def get_reasons(input_vals):
    reasons = []
    for label, (low, high, unit, key) in NORMAL_RANGES.items():
        val = round(input_vals[key], 2)
        unit_str = f" {unit}" if unit else ""
        if val > high:
            reasons.append(f"<b>{label}</b> is {val}{unit_str} — above normal range ({low}–{high}{unit_str})")
        elif val < low:
            reasons.append(f"<b>{label}</b> is {val}{unit_str} — below normal range ({low}–{high}{unit_str})")
    return reasons

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
    tot_bilirubin    = st.number_input('Total Bilirubin (mg/dL)',    min_value=0.0, value=1.0,  step=0.1)
    tot_proteins     = st.number_input('Total Proteins (U/L)',       min_value=0,   value=200)
    ag_ratio         = st.number_input('AG Ratio',                   min_value=0,   value=1)
    alkphos          = st.number_input('Alkaline Phosphotase (U/L)', min_value=0.0, value=0.9,  step=0.1)
with col2:
    direct_bilirubin = st.number_input('Direct Bilirubin (mg/dL)',  min_value=0.0, value=0.3,  step=0.1)
    albumin          = st.number_input('Albumin (g/dL)',             min_value=0,   value=35)
    sgpt             = st.number_input('SGPT (U/L)',                 min_value=0.0, value=5.0,  step=0.1)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button('Run Prediction'):
    name_display = patient_name.strip() if patient_name.strip() else "Patient"

    input_vals = {
        'tot_bilirubin':    tot_bilirubin,
        'direct_bilirubin': direct_bilirubin,
        'tot_proteins':     tot_proteins,
        'albumin':          albumin,
        'ag_ratio':         ag_ratio,
        'sgpt':             sgpt,
        'alkphos':          alkphos,
    }

    # Apply log transformation to skewed features (same as training)
    log_features = ['tot_bilirubin', 'direct_bilirubin', 'tot_proteins',
                    'albumin', 'ag_ratio']

    input_df = pd.DataFrame([[
        age,
        np.log1p(tot_bilirubin),
        np.log1p(direct_bilirubin),
        np.log1p(tot_proteins),
        np.log1p(albumin),
        np.log1p(ag_ratio),
        sgpt,
        alkphos
    ]], columns=['age', 'tot_bilirubin', 'direct_bilirubin',
                 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'alkphos'])

    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[:, 1][0]
    prediction   = 1 if probability >= 0.45 else 0
    reasons      = get_reasons(input_vals)

    st.markdown("---")
    st.markdown(f"### Results for **{name_display}**")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box-sick">
            <h2>⚠️ HIGH RISK — Liver Disease Detected</h2>
            <p style="color:#555; margin-top:0.3rem;">Liver Disease Probability: <strong>{probability:.1%}</strong></p>
            <p style="margin-top:0.8rem; color:#2d2d2d;">Based on the blood test values entered, this patient shows markers consistent with liver disease.</p>
        </div>
        """, unsafe_allow_html=True)

        if reasons:
            st.markdown("**Values outside normal range that contributed to this prediction:**")
            for text in reasons:
                st.markdown(f'<div class="reason-item-red">🔴 {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="reason-item-red">⚠️ Values appear within range but overall pattern indicates elevated risk. Refer for further evaluation.</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box-healthy">
            <h2>✅ LOW RISK — No Liver Disease Detected</h2>
            <p style="color:#555; margin-top:0.3rem;">Liver Disease Probability: <strong>{probability:.1%}</strong></p>
            <p style="margin-top:0.8rem; color:#2d2d2d;">Based on the blood test values entered, this patient's markers are largely within normal range.</p>
        </div>
        """, unsafe_allow_html=True)

        if reasons:
            st.markdown("**Some values were outside normal range but not enough to indicate disease:**")
            for text in reasons:
                st.markdown(f'<div class="reason-item-yellow">🟡 {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown("✅ All values are within normal range.")

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for screening purposes only and does
        not constitute a medical diagnosis. Predictions are based on a dataset of 583 patients
        and may not generalise to all populations. Always consult a qualified medical
        professional for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)