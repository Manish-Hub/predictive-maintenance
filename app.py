import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model and threshold from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ManishTK44/predictive-maintenance-model",
    filename="xgb_model_final.pkl",
    repo_type="model"
)
threshold_path = hf_hub_download(
    repo_id="ManishTK44/predictive-maintenance-model",
    filename="xgb_threshold.txt",
    repo_type="model"
)
model = joblib.load(model_path)
with open(threshold_path, "r") as f:
    threshold = float(f.read().split("=")[-1].strip())

# Try loading scaler (fallback if missing)
try:
    scaler_path = hf_hub_download(
        repo_id="ManishTK44/predictive-maintenance-model",
        filename="scaler.pkl",
        repo_type="model"
    )
    scaler = joblib.load(scaler_path)
except Exception:
    scaler = None

# Streamlit UI
st.title("⚙️ Predictive Maintenance App")
st.write("Predicts engine condition (Normal vs Warning) using sensor inputs.")

# Sidebar metadata
st.sidebar.header("ℹ️ Model Info")
st.sidebar.write(f"Threshold value: {threshold}")
st.sidebar.write("Model: XGBoost (tuned)")
st.sidebar.write("[View on Hugging Face](https://huggingface.co/ManishTK44/predictive-maintenance-model)")

# Inputs
engine_rpm = st.slider("Engine RPM", min_value=30, max_value=10000, value=500)
lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.5, max_value=20.0, value=1.5)
fuel_pressure = st.number_input("Fuel Pressure", min_value=100, max_value=5000, value=4190)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.5, max_value=20.0, value=0.4)
lub_oil_temp = st.slider("Lub Oil Temp (°C)", min_value=600, max_value=9999, value=9995)
coolant_temp = st.slider("Coolant Temp (°C)", min_value=600, max_value=9999, value=8885)

if st.button("Predict"):
    sample_input = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }])

    # Apply scaler if available
    if scaler:
        sample_scaled = scaler.transform(sample_input)
    else:
        sample_scaled = sample_input

    prob = model.predict_proba(sample_scaled)[0][1]
    prob_str = f"{prob:.2%}"
    label = "Warning" if prob >= threshold else "Normal"
    st.write(f"Predicted probability: {prob:.2f}, Threshold: {threshold}")

    # Styled outputs
    if prob >= threshold:
        st.error(f"⚠️ Engine Warning Detected!\nProbability of warning: {prob_str}")
    elif prob >= 0.3:
        st.warning(f"⚠️ Engine Status: Borderline\nProbability of warning: {prob_str}")
    else:
        st.success(f"✅ Engine Status: Normal\nLow risk detected (Probability: {prob_str})")

