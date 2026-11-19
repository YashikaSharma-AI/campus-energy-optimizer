# ⚡ Smart Campus Energy & Load Forecasting Portal

An enterprise-grade Machine Learning web application designed to predict campus electricity consumption based on ambient temperature, occupancy metrics, and temporal factors. Built using Python, Scikit-Learn, Pandas, and Streamlit.

## 📌 Features
- **Interactive Single Simulation:** Adjust real-time campus variables via sliders to forecast immediate energy demand.
- **Batch CSV Processing:** Upload bulk institutional data files to generate fast automated predictions and download exported results.
- **Model Insights & Explainability:** Dedicated analytics tab showcasing pipeline performance metrics ($R^2$ score, MAE) and a feature importance bar chart explaining model decisions.
- **Trained Pipeline:** Utilizes a `GradientBoostingRegressor` model paired with automated `StandardScaler` transformations.

## 🛠️ Project Structure