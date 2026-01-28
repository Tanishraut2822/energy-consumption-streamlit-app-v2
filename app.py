import os
import requests
import joblib
import streamlit as st
import pandas as pd

# --------------------------------------------------
# GOOGLE DRIVE MODEL CONFIG
# --------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1MFWofVrOsjvggSsbq1QvqTKt9MzuQzgt"
MODEL_PATH = "final_energy_consumption_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained ML model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Energy Consumption Predictor",
    layout="centered"
)

# --------------------------------------------------
# TITLE & INTRO
# --------------------------------------------------
st.title("🔋 Smart Appliance Energy Consumption Predictor")

st.markdown("""
This web application predicts **household appliance energy consumption (Wh)**  
using a **machine learning model trained on real household sensor data**.

### 🎯 Objectives
- Reduce electricity cost
- Improve energy efficiency
- Support smart power grid management
""")

# --------------------------------------------------
# INPUT INFO SECTION
# --------------------------------------------------
st.header("ℹ️ Input Information")

with st.expander("Click here to understand inputs"):
    st.markdown("""
**Lights Energy (Wh)**  
Energy consumed by lighting fixtures.

**Average Indoor Temperature (°C)**  
Affects AC/heater usage.

**Average Indoor Humidity (%)**  
Higher humidity increases cooling load.

**Hour of Day (0–23)**  
Energy usage varies by time.

**Day / Month / Weekday**  
Captures daily and seasonal patterns.

Weekday values:
- 0 → Monday
- 6 → Sunday
""")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.header("📥 Enter Household Conditions")

lights = st.number_input("💡 Lights Energy (Wh)", min_value=0.0, value=20.0)
avg_temp = st.number_input("🌡️ Average Indoor Temperature (°C)", value=22.0)
avg_humidity = st.number_input("💧 Average Indoor Humidity (%)", value=45.0)

hour = st.slider("⏰ Hour of Day", 0, 23, 19)
day = st.slider("📅 Day of Month", 1, 31, 15)
month = st.slider("🗓️ Month", 1, 12, 6)
weekday = st.slider("📆 Weekday (0 = Monday, 6 = Sunday)", 0, 6, 3)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.header("⚙️ Prediction")

if st.button("🔮 Predict Energy Consumption"):

    input_data = pd.DataFrame([{
        "lights": lights,

        # Indoor temperatures
        "T1": avg_temp, "T2": avg_temp, "T3": avg_temp,
        "T4": avg_temp, "T5": avg_temp, "T6": avg_temp,
        "T7": avg_temp, "T8": avg_temp, "T9": avg_temp,

        # Indoor humidity
        "RH_1": avg_humidity, "RH_2": avg_humidity, "RH_3": avg_humidity,
        "RH_4": avg_humidity, "RH_5": avg_humidity, "RH_6": avg_humidity,
        "RH_7": avg_humidity, "RH_8": avg_humidity, "RH_9": avg_humidity,

        # Outdoor defaults
        "T_out": 15,
        "Press_mm_hg": 760,
        "RH_out": 60,
        "Windspeed": 2,
        "Visibility": 40,
        "Tdewpoint": 10,

        # Random vars
        "rv1": 0.5,
        "rv2": 0.5,

        # Time features
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday
    }])

    input_data = input_data[model.feature_names_in_]
    prediction = model.predict(input_data)[0]

    st.success(f"🔋 Predicted Appliance Energy Consumption: **{prediction:.2f} Wh**")

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------
    st.subheader("💡 Energy Optimization Tips")

    if 18 <= hour <= 22:
        st.warning("Peak-hour usage detected. Try shifting heavy appliances.")

    if lights > 30:
        st.info("High lighting usage detected. Use LED lights to save energy.")

    if avg_temp < 18 or avg_temp > 26:
        st.info("Maintain indoor temperature between 20–24°C for efficiency.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn & Streamlit")
