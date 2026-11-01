# Self-Training for Student Risk Prediction
# Install: pip install pandas numpy scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and separate labelled and unlabelled records
df = pd.read_csv("Student_risk.csv")

feature_cols = [
    "LMS_Activities",
    "Assignment_Submission",
    "Attendance",
    "Internal_Marks",
]

has_label = df["Risk_Label"].notna() & (df["Risk_Label"].astype(str).str.strip() != "")
labelled = df[has_label].copy()
unlabelled = df[~has_label].copy()

X_labelled = labelled[feature_cols]
# Encode classes as integers: At Risk=1, Not at Risk=0.
y_labelled = labelled["Risk_Label"].map({"Not at Risk": 0, "At Risk": 1}).astype(int)
X_unlabelled = unlabelled[feature_cols]

print("Labelled records:", len(labelled))
print("Unlabelled records:", len(unlabelled))

# 2. Split labelled data. The test set must remain untouched until evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X_labelled, y_labelled,
    test_size=0.20,
    random_state=42,
    stratify=y_labelled
)

# 3. Train a standard supervised model using labelled training records only.
base_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
base_model.fit(X_train, y_train)

# 4. Train a self-training model using labelled training records and unlabelled pool.
# SelfTrainingClassifier iteratively assigns pseudo-labels to confident examples.
self_training_base = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
self_training_model = SelfTrainingClassifier(
    estimator=self_training_base,
    threshold=0.80,
    criterion="threshold",
    max_iter=10
)

X_combined = pd.concat([X_train, X_unlabelled], ignore_index=True)
y_combined = list(y_train) + [-1] * len(X_unlabelled)

self_training_model.fit(X_combined, y_combined)

# 5. Compare both models on exactly the same held-out labelled test records.
supervised_pred = base_model.predict(X_test)
self_training_pred = self_training_model.predict(X_test)

supervised_accuracy = accuracy_score(y_test, supervised_pred)
self_training_accuracy = accuracy_score(y_test, self_training_pred)

print("\\nSupervised model accuracy:", round(supervised_accuracy * 100, 2), "%")
print("Self-training model accuracy:", round(self_training_accuracy * 100, 2), "%")

print("\\nSupervised model report:")
print(classification_report(
    y_test, supervised_pred,
    labels=[0, 1], target_names=["Not at Risk", "At Risk"],
    zero_division=0
))

print("\\nSelf-training model report:")
print(classification_report(
    y_test, self_training_pred,
    labels=[0, 1], target_names=["Not at Risk", "At Risk"],
    zero_division=0
))

# transduction_ contains original labels plus labels assigned to unlabelled samples.
pseudo_label_count = int((self_training_model.transduction_[len(X_train):] != -1).sum())
print("Unlabelled samples assigned pseudo-labels:", pseudo_label_count)
