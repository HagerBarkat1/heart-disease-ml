import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Config & load model
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="centered")

# Make sure final_model.pkl is in the same folder as this app
model = pickle.load(open("final_model.pkl", "rb"))

# -----------------------------
# Header & Disclaimer
# -----------------------------
st.title("💓 Heart Disease Prediction")
st.markdown(
    "Enter patient data below to estimate the risk of heart disease.\n\n"
    "**⚠️ Disclaimer:** This is an *educational demo tool only*. "
    "It must **NOT** be used for real medical diagnosis."
)

st.markdown("---")

# -----------------------------
# Input area (two columns)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("**Age**", 20, 100, 50, help="Patient age in years.")
    chol = st.slider("**Serum Cholesterol (mg/dl)**", 100, 600, 200, help="Cholesterol level in mg/dl.")

with col2:
    thalach = st.slider("**Max Heart Rate Achieved**", 60, 220, 150, help="Maximum heart rate reached during exercise.")
    oldpeak = st.slider("**ST Depression (oldpeak)**", 0.0, 6.5, 1.0, step=0.1,
                        help="ST depression induced by exercise relative to rest (numeric).")

st.markdown("**Chest Pain Type (cp)** — categorical. RFE selected `cp_4.0` (asymptomatic) so we map below.")
cp_option = st.selectbox(
    "Chest Pain Type (select one)",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 — Typical angina: substernal chest pain, provoked by exertion, relieved by rest or nitroglycerin",
        2: "2 — Atypical angina: chest pain lacking one of the typical features",
        3: "3 — Non-anginal pain: pain not consistent with angina",
        4: "4 — Asymptomatic: no chest pain (silent ischemia)"
    }[x],
    help="Choose the chest pain category. The model uses a `cp_4.0` flag (1 if cp == 4, else 0)."
)

st.markdown("---")

# -----------------------------
# Prepare features (order must match training)
# final_features = ['age', 'chol', 'thalach', 'oldpeak', 'cp_4.0']
# -----------------------------
# compute cp_4.0: 1 if cp_option == 4 else 0
cp_4 = 1 if cp_option == 4 else 0

features = np.array([[age, chol, thalach, oldpeak, cp_4]])

# -----------------------------
# Predict button & result
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):
    try:
        prediction = model.predict(features)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        raise

    # Attempt to get probability if available
    prob = None
    try:
        prob = model.predict_proba(features)[0][1]  # probability of positive class
    except Exception:
        prob = None

    # Result header
    st.subheader("📊 Prediction Result")

    # Visual & textual result
    if prediction[0] == 1:
        st.error("⚠️ Predicted: High risk of Heart Disease")
    else:
        st.success("✅ Predicted: Low risk of Heart Disease")

    # Show probability as a number + progress bar + color zone
    if prob is not None:
        pct = float(prob * 100)
        # show metric
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.metric(label="Risk Probability", value=f"{pct:.1f}%")
        with col_b:
            st.progress(int(pct))

        # color-coded zone explanation
        if prob < 0.30:
            st.info("🟢 Low Risk Zone")
        elif prob < 0.70:
            st.warning("🟡 Medium Risk Zone")
        else:
            st.error("🔴 High Risk Zone")
    else:
        st.info("Model does not provide a probability score for this estimator.")

    # final disclaimer reminder
    st.markdown("---")
    st.warning("⚠️ Educational use only — not a substitute for professional medical advice.")
