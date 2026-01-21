from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="centered")

MODEL_PATH = Path("model") / "wine_cultivar_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "flavanoids",
    "proline",
]

st.title("Wine Cultivar Origin Prediction")
st.write("Enter values for the 6 selected features to predict the cultivar (1, 2, or 3).")

with st.form("prediction_form"):
    alcohol = st.number_input("alcohol", min_value=0.0, format="%.4f")
    malic_acid = st.number_input("malic_acid", min_value=0.0, format="%.4f")
    ash = st.number_input("ash", min_value=0.0, format="%.4f")
    alcalinity_of_ash = st.number_input("alcalinity_of_ash", min_value=0.0, format="%.4f")
    flavanoids = st.number_input("flavanoids", min_value=0.0, format="%.4f")
    proline = st.number_input("proline", min_value=0.0, format="%.4f")

    submitted = st.form_submit_button("Predict Cultivar")

if submitted:
    values = np.array([
        alcohol,
        malic_acid,
        ash,
        alcalinity_of_ash,
        flavanoids,
        proline,
    ]).reshape(1, -1)

    pred = model.predict(values)[0]
    cultivar = int(pred) + 1

    st.success(f"Predicted Cultivar: {cultivar}")
