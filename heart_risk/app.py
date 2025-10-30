# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# -----------------------
# Helper: load artifacts
# -----------------------
MODEL_PATH = "best_xgboost_model.pkl"   # your trained xgboost model file
SCALER_PATH = "scaler.joblib"           # optional: StandardScaler / PowerTransformer saved during training
PREPROC_PATH = "preprocessor.joblib"    # optional: a complete sklearn ColumnTransformer/pipeline (preferred)

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()
preprocessor = load_preprocessor()
scaler = load_scaler()

if model is None:
    st.error(f"Model file not found: {MODEL_PATH}. Put your trained model file here and restart.")
    st.stop()

# If model is XGBoost sklearn wrapper, feature names stored in booster
try:
    expected_features = model.get_booster().feature_names
except Exception:
    # fallback: if not available, try attribute feature_names_in_ (sklearn)
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        expected_features = list(expected_features)
    else:
        # unknown: we'll build features dynamically (risky)
        expected_features = None

# -----------------------
# UI: Title & Info
# -----------------------
st.title("💓 Heart Disease Risk Prediction")
st.markdown("Enter patient details on the left and click **Predict**. This demo uses a tuned XGBoost model trained on the Heart dataset.")

with st.expander("ℹ️ About the features (click to expand)", expanded=False):
    st.markdown("""
    **Feature meanings**
    - **age** — Age of the patient (years).  
    - **sex** — 1 = Male, 0 = Female.  
    - **chest_pain_type** — 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic.  
    - **resting_bp_s** — Resting systolic blood pressure (mm Hg).  
    - **cholesterol** — Serum cholesterol in mg/dL.  
    - **fasting_blood_sugar** — 1 if fasting blood sugar > 120 mg/dL, else 0.  
    - **resting_ecg** — 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.  
    - **max_heart_rate** — Maximum heart rate achieved.  
    - **exercise_angina** — 1 = Yes, 0 = No.  
    - **oldpeak** — ST depression induced by exercise relative to rest.  
    - **st_slope** — 0 = Upsloping, 1 = Flat, 2 = Downsloping, 3 = Other.
    """)

# -----------------------
# Sidebar: Input form
# -----------------------
st.sidebar.header("Patient inputs")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50, help="Age of the patient in years.")
sex = st.sidebar.selectbox("Sex", ("Male", "Female"), help="Biological sex: Male/Female.")
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4], index=0,
                                       help="1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic")
resting_bp_s = st.sidebar.number_input("Resting BP (systolic)", min_value=50, max_value=250, value=120,
                                       help="Resting systolic blood pressure in mm Hg.")
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=200,
                                      help="Serum cholesterol in mg/dL.")
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1], index=0,
                                           help="1 if FBS > 120 mg/dL else 0.")
resting_ecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], index=0,
                                   help="0=Normal, 1=ST-T abnormality, 2=Left ventricular hypertrophy.")
max_heart_rate = st.sidebar.number_input("Max Heart Rate", min_value=50, max_value=250, value=150,
                                         help="Maximum heart rate achieved during exercise.")
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], index=0, help="1 = Yes, 0 = No.")
oldpeak = st.sidebar.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                  help="ST depression induced by exercise relative to rest.")
st_slope = st.sidebar.selectbox("ST Slope", [0, 1, 2, 3], index=1,
                                help="0=Upsloping, 1=Flat, 2=Downsloping, 3=Other")

# -----------------------
# Build input dataframe
# -----------------------
input_dict = {
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    "chest_pain_type": [float(chest_pain_type)],
    "resting_bp_s": [resting_bp_s],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [fasting_blood_sugar],
    "resting_ecg": [float(resting_ecg)],
    "max_heart_rate": [max_heart_rate],
    "exercise_angina": [exercise_angina],
    "oldpeak": [oldpeak],
    "st_slope": [float(st_slope)]
}
input_df = pd.DataFrame(input_dict)

st.write("### Preview of input")
st.dataframe(input_df.T, use_container_width=True)

# -----------------------
# Preprocessing before predict
# -----------------------
# If you saved a preprocessor (ColumnTransformer or a pipeline), prefer it.
if preprocessor is not None:
    # Preprocessor should expect raw columns and output model-ready features
    X_ready = preprocessor.transform(input_df)
    # If preprocessor returns numpy array, convert to DataFrame with expected feature names if available
    if expected_features is not None:
        X_ready = pd.DataFrame(X_ready, columns=expected_features)
else:
    # One-hot encode categorical features exactly like training
    # Categorical features that were one-hot encoded: chest_pain_type, resting_ecg, st_slope
    cat_features = ["chest_pain_type", "resting_ecg", "st_slope"]
    X_enc = pd.get_dummies(input_df, columns=cat_features)
    # Also ensure column types match training (float/ints)
    # If scaler saved, apply it to continuous features
    cont_cols = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
    if scaler is not None:
        try:
            X_enc[cont_cols] = scaler.transform(X_enc[cont_cols])
        except Exception:
            # if scaler expects different columns or shapes, skip with a warning
            st.warning("Loaded scaler could not be applied to input — ensure scaler was saved from training with same columns.")
    else:
        st.info("No preprocessor or scaler found. App will still run but predictions may be less accurate if you trained with scaling or advanced transforms. To remove this message, save your preprocessor as 'preprocessor.joblib' or scaler as 'scaler.joblib' in the app folder.")

    # Align with model's expected features:
    if expected_features is not None:
        # Create missing columns with zeros
        for col in expected_features:
            if col not in X_enc.columns:
                X_enc[col] = 0
        # Reorder to expected ordering
        X_ready = X_enc[expected_features].astype(float)
    else:
        # If we don't have expected_features, use the columns we created (less safe)
        X_ready = X_enc

# -----------------------
# Predict button
# -----------------------
if st.button("Predict"):
    try:
        # If X_ready is DataFrame or array, pass to model
        pred = model.predict(X_ready)
        # probability if available
        prob = None
        try:
            prob = model.predict_proba(X_ready)[:, 1]
        except Exception:
            # some wrappers might use decision_function
            try:
                df_dec = model.decision_function(X_ready)
                # normalize to [0,1]
                prob = (df_dec - df_dec.min()) / (df_dec.max() - df_dec.min() + 1e-8)
            except Exception:
                prob = None

        result = int(pred[0])
        if prob is not None:
            confidence = float(prob[0])
        else:
            confidence = None

        if result == 1:
            st.error(f"⚠️ Prediction: High risk of heart disease (class=1){' — probability: ' + str(round(confidence,3)) if confidence is not None else ''}")
        else:
            st.success(f"✅ Prediction: Low risk of heart disease (class=0){' — probability: ' + str(round(confidence,3)) if confidence is not None else ''}")

        # Optional: show the processed feature vector we fed to the model
        with st.expander("Show model input (processed)"):
            st.dataframe(X_ready.T, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.stop()

# -----------------------
# Footer: diagnostics
# -----------------------
st.markdown("---")
st.caption("Notes: For most accurate predictions, save and place the same preprocessing pipeline used during training as 'preprocessor.joblib' (ColumnTransformer or Pipeline), or a scaler as 'scaler.joblib'. This app will try to align one-hot columns automatically if preprocessor is not present.")
