import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Create synthetic dataset using dictionary
rng = np.random.default_rng(42); n = 150
c1 = rng.normal((-4, 0), (0.22, 0.22), (n, 2))
c2 = rng.normal((0, 0), (0.42, 0.42), (n, 2))
c3 = rng.normal((4, 0), (0.68, 0.68), (n, 2))
noise = rng.uniform((-8, -4), (8, 4), (60, 2))

data = {
    "Feature_1": np.r_[c1[:,0], c2[:,0], c3[:,0], noise[:,0]],
    "Feature_2": np.r_[c1[:,1], c2[:,1], c3[:,1], noise[:,1]],
    "True_Label": np.r_[np.zeros(n,int), np.ones(n,int),
                        np.full(n,2), np.full(len(noise),-1)]
}
df = pd.DataFrame(data)
# df.to_csv("dbscan_dataset.csv", index=False)
df = pd.read_csv("dbscan_dataset.csv")

# 2. Preprocessing
features = ["Feature_1", "Feature_2"]
df.drop_duplicates(inplace=True)
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df[features] = df[features].fillna(df[features].median())
X = StandardScaler().fit_transform(df[features])
y = df["True_Label"].astype(int).to_numpy()

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df[features].isna().sum())

# 3. Find eps using K-distance graph
MIN_SAMPLES = 2 * X.shape[1]
distances = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X).kneighbors(X)[0]
k_distances = np.sort(distances[:, -1])
EPS = 0.30

# 4. Apply DBSCAN
model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
db_labels = model.fit_predict(X)

core = np.zeros(len(X), dtype=bool)
core[model.core_sample_indices_] = True
border = (db_labels != -1) & ~core

print(f"\nDBSCAN Clusters: {len(set(db_labels)-{-1})}")
print("Core Points:", core.sum())
print("Border Points:", border.sum())
print("Noise Points:", (db_labels == -1).sum())

# 5. Compare with other clustering methods
km_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
agg_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)

valid = db_labels != -1

def evaluate(labels):
    sil = silhouette_score(X[valid], labels[valid]) if 1 < len(np.unique(labels[valid])) < valid.sum() else np.nan
    return sil, adjusted_rand_score(y, labels)

results = []
for name, labels in [("DBSCAN", db_labels), ("K-Means", km_labels),
                     ("Agglomerative", agg_labels)]:
    sil, ari = evaluate(labels)
    results.append([name, len(set(labels)-{-1}), (labels == -1).sum(), sil, ari])

print("\nPerformance Comparison:")
print(pd.DataFrame(results, columns=["Method", "Clusters", "Noise",
      "Silhouette", "ARI"]).round(3).to_string(index=False))

# 6. Visualizations
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax[0,0].plot(k_distances)
ax[0,0].axhline(EPS, color="red", linestyle="--", label=f"eps={EPS}")
ax[0,0].set(title="K-distance Graph", xlabel="Points (sorted)",
            ylabel="Distance to 3rd neighbour")
ax[0,0].legend()

for a, labels, title in zip(ax.flat[1:],
        [db_labels, km_labels, agg_labels],
        ["DBSCAN", "K-Means", "Agglomerative"]):
    a.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=18)
    a.set(title=title, xlabel="Scaled Feature 1", ylabel="Scaled Feature 2")

plt.tight_layout()
plt.show()