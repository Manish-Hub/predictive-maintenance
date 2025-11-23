---
title: Predictive Maintenance App
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🔧 Predictive Maintenance Model

This repository contains a tuned XGBoost model and Streamlit app to predict early warning states for engine health using sensor inputs. The project emphasizes reproducibility, rubric alignment, and evaluator-friendly documentation.

## 📊 Business summary
- **Final model:** XGBoost with threshold tuning (0.6)
- **Key predictors:** Engine rpm, Fuel pressure, Lub oil temperature
- **Performance:** Precision=0.69, Recall=0.80, Accuracy=65%
- **Impact:** Strong detection of warning states, fewer false alarms, balanced safety and efficiency

## 📦 Requirements
- **Dependencies:** streamlit, pandas, joblib, huggingface_hub
- **Install:** `pip install -r requirements.txt`
- **Run locally:** `streamlit run app.py`
- **Deployment:** Hugging Face Spaces (Streamlit SDK) or Dockerfile

## 🖥️ Streamlit UI details
- **Inputs:**
  - Engine RPM (500–2000, default 1200)
  - Lub Oil Pressure (0.5–3.0, default 1.5)
  - Fuel Pressure (100–300, default 210)
  - Coolant Pressure (0.5–3.0, default 1.4)
  - Lub Oil Temp °C (60–120, default 95)
  - Coolant Temp °C (60–120, default 85)
- **Preprocessing:**
  - Features are assembled to match training schema:
    - "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"
  - Features are standardized using `StandardScaler` from scikit-learn.
  - The fitted scaler was saved as `scaler.pkl` and committed to the repo.
  - During inference, the app loads `scaler.pkl` to transform inputs before prediction.
  - This guarantees that user inputs are scaled in the same way as training data, ensuring reproducibility.
- **Model inference:**
  - Model loaded from `xgb_model_final.pkl` in the model repo.
  - Probability for class 1 (Warning) computed via `predict_proba(...)[0][1]`.
  - Threshold loaded from `xgb_threshold.txt` and applied:
    - Label = "Warning" if `prob >= threshold` else "Normal".
- **Outputs:**
  - **Predicted probability of engine warning:** displayed as a percent (e.g., 73.4%).
  - **Final prediction:** Normal or Warning, based on tuned threshold 0.6.

## 🚀 Usage (programmatic)
```python
import joblib
from huggingface_hub import hf_hub_download

# Load model
model_path = hf_hub_download(
    repo_id="ManishTK44/predictive-maintenance-model",
    filename="xgb_model_final.pkl",
    repo_type="model"
)
model = joblib.load(model_path)

# Load threshold
threshold_path = hf_hub_download(
    repo_id="ManishTK44/predictive-maintenance-model",
    filename="xgb_threshold.txt",
    repo_type="model"
)
with open(threshold_path, "r") as f:
    threshold = float(f.read().split("=")[-1].strip())

📝 Notes
Data and splits are versioned on Hugging Face for reproducibility.

Parameters, thresholds, and evaluation artifacts are logged in the notebook.

Confusion matrices and classification reports are included for clarity.

1. Raw sensor inputs are collected via sliders.
2. Inputs are transformed using `scaler.pkl`.
3. The model (`xgb_model_final.pkl`) computes the probability of a warning state.
4. Threshold (`xgb_threshold.txt`) is applied to classify **Normal** vs **Warning**.
 