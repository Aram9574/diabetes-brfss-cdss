"""
Diabetes Risk Stratification — CDC BRFSS CDSS
Author: Alejandro Zakzuk | Physician · AI Applied to Health
---
Multiclass risk stratification: No diabetes / Prediabetes / Diabetes
Dataset: CDC BRFSS 2015 · 253,680 U.S. adults
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

st.set_page_config(
    page_title="Diabetes Risk CDSS — BRFSS",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size:1.9rem; font-weight:700; color:#1a1a2e; }
.subtitle { font-size:0.95rem; color:#666; margin-bottom:1rem; }
.risk-high   { background:#fde8e8; border-left:5px solid #e53e3e; padding:1rem; border-radius:6px; }
.risk-medium { background:#fef3cd; border-left:5px solid #d69e2e; padding:1rem; border-radius:6px; }
.risk-low    { background:#e6f4ea; border-left:5px solid #38a169; padding:1rem; border-radius:6px; }
.risk-predn  { background:#fff3e0; border-left:5px solid #FF8C00; padding:1rem; border-radius:6px; }
.warning-box { background:#fff8e1; border:1px solid #f59e0b; padding:0.7rem; border-radius:6px; font-size:0.85rem; }
.disclaimer  { background:#f0f4ff; border:1px solid #c3d0f0; padding:0.8rem; border-radius:6px; font-size:0.82rem; color:#444; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base, '..', 'src')
    model     = joblib.load(os.path.join(src, 'model_xgb.pkl'))
    explainer = joblib.load(os.path.join(src, 'shap_explainer.pkl'))
    scaler    = joblib.load(os.path.join(src, 'scaler.pkl'))
    features  = joblib.load(os.path.join(src, 'feature_names.pkl'))
    return model, explainer, scaler, features

try:
    model, explainer, scaler, feature_names = load_artifacts()
    loaded = True
except Exception as e:
    loaded = False
    st.error(f"Model artifacts not found. Run notebooks 01–03 first.\n{e}")

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.markdown('<p class="main-title">🩺 Diabetes Risk Stratification — CDSS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">CDC BRFSS 2015 · 253,680 U.S. adults · XGBoost · Multiclass (No DM / Prediabetes / Diabetes)</p>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<div style="background:#f8f9fa;border-radius:8px;padding:0.7rem;text-align:center"><div style="font-size:1.3rem;font-weight:700;color:#2d6a4f;">84.2%</div><div style="font-size:0.75rem;color:#666">No DM class accuracy</div></div>', unsafe_allow_html=True)
with col_h3:
    st.markdown('<div style="background:#f8f9fa;border-radius:8px;padding:0.7rem;text-align:center"><div style="font-size:1.3rem;font-weight:700;color:#c44e52;">253k</div><div style="font-size:0.75rem;color:#666">Training records</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
⚠️ <strong>Known limitation:</strong> Prediabetes detection is unreliable with survey-only data (no HbA1c or fasting glucose). 
This model can suggest whether someone may be <em>at risk for diabetes</em>, but cannot reliably distinguish prediabetes from normoglycemia 
without laboratory values. This is documented as a clinical finding, not a modeling failure.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if not loaded:
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📋 Survey Inputs")
st.sidebar.markdown("Respond to each item as you would in a health survey.")
st.sidebar.markdown("---")

def yn(label, help_text=None):
    return st.sidebar.selectbox(label, ['No', 'Yes'], help=help_text)

def yn_val(response): return 1.0 if response == 'Yes' else 0.0

st.sidebar.markdown("**Medical history**")
high_bp   = yn("High blood pressure ever diagnosed?")
high_chol = yn("High cholesterol ever diagnosed?")
chol_check= yn("Cholesterol check in past 5 years?")
stroke    = yn("Ever had a stroke?")
heart     = yn("Heart disease or heart attack?")

st.sidebar.markdown("---")
st.sidebar.markdown("**Lifestyle**")
smoker    = yn("Smoked at least 100 cigarettes in your life?")
phys_act  = yn("Physical activity in past 30 days (outside work)?")
fruits    = yn("Consume fruit 1 or more times per day?")
veggies   = yn("Consume vegetables 1 or more times per day?")
heavy_alc = yn("Heavy alcohol consumption?", "Men: >14 drinks/week | Women: >7 drinks/week")

st.sidebar.markdown("---")
st.sidebar.markdown("**Health status**")
bmi       = st.sidebar.slider("BMI", 10.0, 70.0, 27.0, 0.5)
gen_hlth  = st.sidebar.select_slider("General health (1=Excellent, 5=Poor)",
                                      options=[1,2,3,4,5], value=3)
ment_hlth = st.sidebar.slider("Days of poor mental health (past 30 days)", 0, 30, 0)
phys_hlth = st.sidebar.slider("Days of poor physical health (past 30 days)", 0, 30, 0)
diff_walk = yn("Difficulty walking or climbing stairs?")
any_hc    = yn("Any kind of health care coverage?")
no_doc    = yn("Couldn't see doctor in past year due to cost?")

st.sidebar.markdown("---")
st.sidebar.markdown("**Demographics**")
sex       = st.sidebar.selectbox("Sex", ["Female", "Male"])
age_cat   = st.sidebar.selectbox("Age group", [
    "18–24", "25–29", "30–34", "35–39", "40–44",
    "45–49", "50–54", "55–59", "60–64", "65–69",
    "70–74", "75–79", "80+"
])
education = st.sidebar.select_slider("Education level",
    options=[1,2,3,4,5,6],
    format_func=lambda x: {1:"No school",2:"Elementary",3:"Some HS",
                            4:"HS grad",5:"Some college",6:"College grad"}[x], value=4)
income    = st.sidebar.select_slider("Annual household income",
    options=[1,2,3,4,5,6,7,8],
    format_func=lambda x: {1:"<$10k",2:"$10–15k",3:"$15–20k",4:"$20–25k",
                            5:"$25–35k",6:"$35–50k",7:"$50–75k",8:">$75k"}[x], value=5)

predict_btn = st.sidebar.button("🔍 Assess Risk", use_container_width=True, type="primary")

# ── Predict ───────────────────────────────────────────────────────────────────
age_map = {"18–24":1,"25–29":2,"30–34":3,"35–39":4,"40–44":5,
           "45–49":6,"50–54":7,"55–59":8,"60–64":9,"65–69":10,
           "70–74":11,"75–79":12,"80+":13}

if predict_btn:
    patient = pd.DataFrame([{
        'HighBP':                yn_val(high_bp),
        'HighChol':              yn_val(high_chol),
        'CholCheck':             yn_val(chol_check),
        'BMI':                   float(bmi),
        'Smoker':                yn_val(smoker),
        'Stroke':                yn_val(stroke),
        'HeartDiseaseorAttack':  yn_val(heart),
        'PhysActivity':          yn_val(phys_act),
        'Fruits':                yn_val(fruits),
        'Veggies':               yn_val(veggies),
        'HvyAlcoholConsump':     yn_val(heavy_alc),
        'AnyHealthcare':         yn_val(any_hc),
        'NoDocbcCost':           yn_val(no_doc),
        'GenHlth':               float(gen_hlth),
        'MentHlth':              float(ment_hlth),
        'PhysHlth':              float(phys_hlth),
        'DiffWalk':              yn_val(diff_walk),
        'Sex':                   1.0 if sex == "Male" else 0.0,
        'Age':                   float(age_map[age_cat]),
        'Education':             float(education),
        'Income':                float(income),
    }])[feature_names]

    patient_sc = pd.DataFrame(scaler.transform(patient), columns=feature_names)
    probs = model.predict_proba(patient_sc)[0]
    pred_class = int(np.argmax(probs))

    class_info = {
        0: ("🟢 LOW RISK", "risk-low",   "No diabetes suggested"),
        1: ("🟡 MONITOR", "risk-predn", "Prediabetes range — note: low model reliability without lab values"),
        2: ("🔴 HIGH RISK", "risk-high",  "Diabetes range — clinical evaluation recommended"),
    }
    label, css_class, subtitle = class_info[pred_class]

    col_res, col_prob, col_shap = st.columns([1.2, 0.8, 2])

    with col_res:
        st.subheader("Risk Assessment")
        st.markdown(f"""
        <div class="{css_class}">
            <h2 style="margin:0">{label}</h2>
            <p style="margin:0.3rem 0 0; font-size:0.9rem">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            'Class': ['No DM', 'Prediabetes', 'Diabetes'],
            'Probability': probs
        })
        fig_pb, ax_pb = plt.subplots(figsize=(4, 2.5))
        colors = ['#4C72B0', '#FF8C00', '#C44E52']
        bars = ax_pb.barh(prob_df['Class'], prob_df['Probability'],
                          color=colors, alpha=0.85)
        for bar, val in zip(bars, probs):
            ax_pb.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                      f'{val:.1%}', va='center', fontsize=9)
        ax_pb.set_xlim(0, 1.15)
        ax_pb.set_xlabel('Probability')
        ax_pb.set_title('Model Output', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pb)
        plt.close()

    with col_shap:
        st.subheader("Feature Contributions — This Patient")
        shap_vals = explainer.shap_values(patient_sc)
        sv = shap_vals[0, :, pred_class]  # shape (21,) for predicted class

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP': sv,
            'Value': patient.values[0]
        }).sort_values('SHAP', key=abs, ascending=True).tail(12)

        fig_sh, ax_sh = plt.subplots(figsize=(7, 5))
        colors_sh = ['#e53e3e' if v > 0 else '#38a169' for v in shap_df['SHAP']]
        bars_sh = ax_sh.barh(shap_df['Feature'], shap_df['SHAP'],
                              color=colors_sh, alpha=0.85, edgecolor='white')
        ax_sh.axvline(0, color='black', linewidth=0.8)
        ax_sh.set_xlabel(f'SHAP contribution to [{["No DM","Prediabetes","Diabetes"][pred_class]}] probability')
        ax_sh.set_title('What drove this prediction?', fontweight='bold', fontsize=11)
        for bar, (_, row) in zip(bars_sh, shap_df.iterrows()):
            offset = 0.001 if row['SHAP'] >= 0 else -0.001
            ha = 'left' if row['SHAP'] >= 0 else 'right'
            ax_sh.text(row['SHAP'] + offset, bar.get_y() + bar.get_height()/2,
                       f"{row['Value']:.0f}", va='center', ha=ha, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_sh)
        plt.close()

        st.markdown("🔴 Red = increases risk &nbsp;|&nbsp; 🟢 Green = decreases risk &nbsp;|&nbsp; Numbers = patient value")

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Clinical Disclaimer:</strong> This tool is for educational and research purposes only. It does not constitute 
    medical advice and is not validated for clinical use. Prediabetes and diabetes diagnosis requires biochemical 
    confirmation (HbA1c or fasting plasma glucose) per ADA 2024 guidelines. 
    The prediabetes class prediction specifically is unreliable with survey-only data — do not use it as a screening tool.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Fill in the survey in the sidebar and click **Assess Risk**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Dataset**\n- CDC BRFSS 2015\n- 253,680 U.S. adults\n- 3-class target\n- 21 survey features")
    with c2:
        st.markdown("**Model**\n- XGBoost (winner)\n- Targeted SMOTE\n- Stratified 5-fold CV\n- Balanced accuracy metric")
    with c3:
        st.markdown("**Explainability**\n- SHAP TreeExplainer\n- Global + local\n- Per-class contributions\n- Equity analysis included")
    st.markdown("---")
    st.markdown("**[GitHub](https://github.com/Aram9574/diabetes-brfss-cdss)** · **[Author](https://alejandrozakzuk.com)**")
