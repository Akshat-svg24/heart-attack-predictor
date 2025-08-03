
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🫀 Heart Attack Risk Predictor")
st.write("Enter your health data below to assess your risk of a heart attack.")

# User inputs
age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

# Predict
if st.button("Predict Risk"):
    input_data = pd.DataFrame([[
        age, 1 if sex == "Male" else 0, cp, trestbps, chol,
        fbs, restecg, thalach, exang
    ]], columns=["age", "sex", "cp", "trestbps", "chol",
                 "fbs", "restecg", "thalach", "exang"])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart attack. Please consult a doctor immediately.")
    else:
        st.success("✅ Low risk. Keep maintaining a healthy lifestyle!")
