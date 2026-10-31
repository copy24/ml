import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create dataset manually, save as CSV, then import it
data = {"Study_Hours": [1,1.5,2,2.5,3.0,3.5,4,None,5,5.5,6,6.5,7,7.5,8],
        "Marks": [35,40,42,46,49,53,55,60,63,None,70,73,77,80,85]}
# pd.DataFrame(data).to_csv("linear_regression_dataset.csv", index=False)
df = pd.read_csv("linear_regression_dataset.csv")

print("Dataset:\n", df, "\n\nMissing values:\n", df.isnull().sum())
df = df.dropna()
X, y = df[["Study_Hours"]], df["Marks"]

# Split data, train model and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nActual:", y_test.to_list(), "\nPredicted:", np.round(y_pred, 2).tolist())
print(f"\nEquation: Marks = {model.coef_[0]:.2f} * Study_Hours + {model.intercept_:.2f}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE  = {mse:.2f}\nRMSE = {np.sqrt(mse):.2f}")
print(f"MAE  = {mean_absolute_error(y_test, y_pred):.2f}\nR2   = {r2_score(y_test, y_pred):.4f}")

# Plot actual data points and regression line
x_line = pd.DataFrame({"Study_Hours": np.linspace(X.Study_Hours.min(), X.Study_Hours.max(), 100)})
plt.scatter(X["Study_Hours"], y, color="blue", label="Actual data")
plt.plot(x_line["Study_Hours"], model.predict(x_line), color="red", label="Regression line")
plt.xlabel("Study Hours"); plt.ylabel("Marks")
plt.title("Linear Regression: Study Hours vs Marks")
plt.legend(); plt.grid(True); plt.show()