import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("model", exist_ok=True)

np.random.seed(42)
n_samples = 2000

hours = np.random.randint(0, 24, n_samples)
days_of_week = np.random.randint(0, 7, n_samples)
temperatures = np.random.uniform(15.0, 42.0, n_samples)
occupancy_levels = np.random.randint(50, 1500, n_samples)
is_weekend = (days_of_week >= 5).astype(int)

base_load = 200
energy_load = (
    base_load
    + (occupancy_levels * 0.35)
    + (np.maximum(0, temperatures - 25) * 12.5)
    + (np.sin(hours * np.pi / 12) * 150)
    - (is_weekend * 180)
    + np.random.normal(0, 25, n_samples)
)

df = pd.DataFrame({
    "hour": hours,
    "day_of_week": days_of_week,
    "temperature": temperatures,
    "occupancy_level": occupancy_levels,
    "is_weekend": is_weekend,
    "energy_load_kwh": energy_load,
})

X = df.drop(columns=["energy_load_kwh"])
y = df["energy_load_kwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, random_state=42
    ),
}

print("Evaluating candidate models...")
for name, reg in models.items():
  pipe = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
  pipe.fit(X_train, y_train)
  preds = pipe.predict(X_test)
  print(
      f"{name} -> R²: {r2_score(y_test, preds):.4f} | MAE:"
      f" {mean_absolute_error(y_test, preds):.2f} kWh"
  )

# Train and save the winning Gradient Boosting pipeline
final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "regressor",
        GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, random_state=42
        ),
    ),
])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, "model/energy_model.pkl")
print("\nWinning model (Gradient Boosting) saved to model/energy_model.pkl")
