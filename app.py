import streamlit as st
import pandas as pd
import numpy as np
import joblib




st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)




model = joblib.load("xgboost_model.pkl")

label_encoders = joblib.load(
    "label_encoders.pkl"
)




st.title("🌾 Crop Yield Prediction System")

st.markdown("""
Predict agricultural crop yield using:

- Crop Type
- Rainfall Data
- Fertilizer Usage
- Pesticide Usage
- Temperature Conditions
- Seasonal Information

Built using XGBoost Regression.
""")




crop = st.selectbox(
    "Select Crop",
    label_encoders['crop'].classes_
)

season = st.selectbox(
    "Select Season",
    label_encoders['season'].classes_
)

state = st.selectbox(
    "Select State",
    label_encoders['state'].classes_
)

crop_year = st.number_input(
    "Crop Year",
    min_value=2000,
    max_value=2035,
    value=2024
)

area = st.number_input(
    "Cultivation Area",
    min_value=1.0,
    value=100.0
)

annual_rainfall = st.number_input(
    "Annual Rainfall (mm)",
    min_value=0.0,
    value=1200.0
)

fertilizer = st.number_input(
    "Fertilizer Usage",
    min_value=0.0,
    value=500.0
)

pesticide = st.number_input(
    "Pesticide Usage",
    min_value=0.0,
    value=50.0
)

avg_temperature = st.number_input(
    "Average Temperature (°C)",
    value=25.0
)

max_temperature = st.number_input(
    "Maximum Temperature (°C)",
    value=32.0
)

min_temperature = st.number_input(
    "Minimum Temperature (°C)",
    value=18.0
)




if st.button("Predict Yield"):

    # =====================================
    # FEATURE ENGINEERING
    # =====================================

    temp_range = (
        max_temperature -
        min_temperature
    )

    rainfall_intensity = (
        annual_rainfall / 12
    )

    fertilizer_per_area = (
        fertilizer / (area + 1)
    )

    pesticide_per_area = (
        pesticide / (area + 1)
    )

    area_log = np.log1p(area)

    years_from_2000 = (
        crop_year - 2000
    )


    crop_encoded = (
        label_encoders['crop']
        .transform([crop])[0]
    )

    season_encoded = (
        label_encoders['season']
        .transform([season])[0]
    )

    state_encoded = (
        label_encoders['state']
        .transform([state])[0]
    )

  

    input_df = pd.DataFrame({

        'crop': [crop_encoded],

        'crop_year': [crop_year],

        'season': [season_encoded],

        'state': [state_encoded],

        'area': [area],

        'annual_rainfall': [annual_rainfall],

        'fertilizer': [fertilizer],

        'pesticide': [pesticide],

        'avg_temperature': [avg_temperature],

        'max_temperature': [max_temperature],

        'min_temperature': [min_temperature],

        'temp_range': [temp_range],

        'rainfall_intensity': [
            rainfall_intensity
        ],

        'fertilizer_per_area': [
            fertilizer_per_area
        ],

        'pesticide_per_area': [
            pesticide_per_area
        ],

        'area_log': [area_log],

        'years_from_2000': [
            years_from_2000
        ]
    })

  

    prediction_log = model.predict(
        input_df
    )

    prediction = np.expm1(
        prediction_log
    )

    predicted_yield = prediction[0]

  

    st.success(
        f"""
        🌾 Estimated Crop Yield

        ## {predicted_yield:.2f} tonnes/hectare
        """
    )



    if predicted_yield < 2:

        st.warning(
            "⚠️ Low predicted agricultural productivity."
        )

    elif predicted_yield < 5:

        st.info(
            "ℹ️ Moderate predicted agricultural productivity."
        )

    else:

        st.success(
            "✅ High predicted agricultural productivity."
        )

  

    st.markdown("---")

    st.subheader("📊 Prediction Insights")

    st.write(
        f"• Rainfall Intensity: "
        f"{rainfall_intensity:.2f}"
    )

    st.write(
        f"• Temperature Range: "
        f"{temp_range:.2f} °C"
    )

    st.write(
        f"• Fertilizer per Area: "
        f"{fertilizer_per_area:.2f}"
    )

    st.write(
        f"• Pesticide per Area: "
        f"{pesticide_per_area:.2f}"
    )

   

    st.balloons()