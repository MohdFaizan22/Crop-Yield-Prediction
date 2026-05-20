# 🌾 Crop Yield Prediction using XGBoost

## 📌 Overview

This project predicts agricultural crop yield using machine learning and environmental factors such as rainfall, temperature, fertilizer usage, pesticide usage, and crop-specific information.

The system was built using XGBoost Regression and deployed using Streamlit to provide real-time yield prediction.

---

# 🚀 Problem Statement

Agriculture is highly dependent on weather conditions, farming practices, and resource management. Farmers often face uncertainty regarding crop productivity due to unpredictable rainfall patterns, climate variations, and inefficient resource utilization.

This project aims to solve this problem by building a machine learning-based crop yield prediction system using historical agricultural and climate-related data.

The model helps estimate expected crop yield before cultivation completion, enabling:

- Better agricultural planning
- Improved fertilizer management
- Resource optimization
- Climate-aware farming decisions
- Risk reduction for farmers

---

# 📂 Dataset Features

| Feature         |   Description         |
|-----------------|-----------------------|
| Crop            | Crop type             |
| Crop_Year       | Year of cultivation   |
| Season          | Agricultural season   |
| State           | Indian state          |
| Area            | Cultivated area       |
| Annual_Rainfall | Total annual rainfall |
| Fertilizer      | Fertilizer usage      |
| Pesticide       | Pesticide usage       |
| Avg_Temperature | Average temperature   |
| Max_Temperature | Maximum temperature   |
| Min_Temperature | Minimum temperature   |
| Yield           | Target variable       |

---

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

# 🧠 Machine Learning Workflow

## 1. Data Cleaning
- Removed outliers
- Standardized categorical columns
- Applied log transformation on target variable

---

## 2. Feature Engineering

Created additional features:
- Temperature Range
- Rainfall Intensity
- Fertilizer per Area
- Pesticide per Area
- Log-transformed Area
- Years from 2000

---

## 3. Model Training

Trained and compared:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

# 📊 Model Performance

| Model | R² Score |
|---|---|
| Linear Regression | 0.15 |
| Random Forest | 0.91 |
| XGBoost | 0.94 |

XGBoost achieved the best performance.

---

# 📈 Key Findings

- Agricultural yield relationships are highly nonlinear.
- Tree-based ensemble models outperform linear models significantly.
- Rainfall and temperature strongly influence crop productivity.
- Feature engineering greatly improved model accuracy.
- XGBoost captured complex agricultural patterns effectively.

---

# 🌐 Streamlit Application

The Streamlit app allows users to:
- Select crop type
- Enter rainfall and temperature conditions
- Provide fertilizer and pesticide usage
- Predict crop yield instantly

---

# ▶️ Run Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit app

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
crop-yield-prediction/
│
├── app.py
├── xgboost_model.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
├── crop_yield_analysis.ipynb
└── dataset.csv
```

---

# 🔮 Future Improvements

- Real-time weather API integration
- District-level prediction
- GeoPandas choropleth visualization
- SHAP explainability dashboard
- Satellite imagery integration

---

# 👨‍💻 Author

Mohd Faizanullah

Machine Learning | Data Science | AgriTech