import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------
# 1. Load and Prepare the Dataset
# --------------------------------------------------------
# fetch_openml returns the target as string, so we convert it to float
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data.apply(pd.to_numeric, errors="coerce")       # ensure all features are numeric
y = pd.to_numeric(boston.target, errors="coerce")           # convert target to numeric

print("✅ Data Loaded Successfully")
print("Dataset Shape:", X.shape)
print("\nFirst 5 Rows:\n", X.head())

# --------------------------------------------------------
# 2. Split the Data
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# 3. Train a Linear Regression Model
# --------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------------
# 4. Make Predictions
# --------------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------------
# 5. Evaluate the Model
# --------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Squared Error : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R² Score : {r2:.2f}")

# --------------------------------------------------------
# 6. Visualization – Actual vs Predicted
# --------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", edgecolor="k", alpha=0.7)
plt.xlabel("Actual Prices (in $1000s)")
plt.ylabel("Predicted Prices (in $1000s)")
plt.title("Actual vs Predicted House Prices")

# Optional: add a diagonal reference line
line_coords = np.linspace(min(y_test), max(y_test))
plt.plot(line_coords, line_coords, color="red", linewidth=2, label="Ideal Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

