import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Create dataset manually using dictionary
data = {
    "Age": [20,22,np.nan,25,27,29,21,23,26,28,30,19,35,38,40,42,45,47,36,39,43,44,41,46,48,50,52,55,58,60,62,64,53,57,66,59],
    "Annual_Income": [22,25,28,30,32,35,24,27,31,34,29,26,55,60,65,np.nan,70,75,58,63,67,72,62,69,95,100,105,110,115,120,125,130,108,118,135,112],
    "Spending_Score": [82,78,88,85,75,80,90,84,77,87,83,79,45,52,50,58,48,55,42,60,53,47,57,51,20,25,18,30,np.nan,28,15,24,32,19,12,27],
    "Membership_Type": ["Basic","Basic","Basic","Basic","Basic","Basic","Basic",np.nan,"Basic","Basic","Basic","Basic","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Standard","Premium","Premium","Premium","Premium","Premium","Premium","Premium","Premium","Premium","Premium","Premium","Premium"]
}

df = pd.DataFrame(data)
df.to_csv("kmeans_customer_dataset.csv", index=False)

print("Original Dataset:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 2. Separate numerical and categorical columns
num_cols = ["Age", "Annual_Income", "Spending_Score"]
cat_cols = ["Membership_Type"]

# 3. Preprocessing: imputation, scaling and encoding
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X = preprocessor.fit_transform(df)

print("\nPreprocessed Dataset Shape:", X.shape)

# 4. Apply K-Means for different values of K
results = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X)
    wcss = model.inertia_
    sil = silhouette_score(X, labels)
    results.append([k, wcss, sil])

# 5. Compare clustering results
results_df = pd.DataFrame(results, columns=["K", "WCSS", "Silhouette Score"])
print("\nK-Means Evaluation Results:\n")
print(results_df.round(3).to_string(index=False))

# 6. Select optimal K using highest Silhouette Score
best_k = int(results_df.loc[results_df["Silhouette Score"].idxmax(), "K"])
print("\nOptimal Number of Clusters (K):", best_k)

# 7. Train final K-Means model
final_model = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = final_model.fit_predict(X)

print("\nFinal Cluster Assignments:\n")
print(df.to_string(index=False))

print("\nNumber of Points in Each Cluster:\n")
print(df["Cluster"].value_counts().sort_index())

# 8. Plot Elbow Method and Silhouette Score
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(results_df["K"], results_df["WCSS"], marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of Clusters (K)")
ax[0].set_ylabel("WCSS (Inertia)")
ax[0].grid()

ax[1].plot(results_df["K"], results_df["Silhouette Score"], marker="o")
ax[1].axvline(best_k, linestyle="--", label=f"Optimal K = {best_k}")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of Clusters (K)")
ax[1].set_ylabel("Silhouette Score")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()