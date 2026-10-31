import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Create dataset manually using dictionary
data = {
    "Student_ID": list(range(1, 41)),
    "Age": [18,19,20,21,22,23,19,20,21,22,18,24,19,20,22,23,18,21,20,24,19,22,23,18,21,20,24,19,22,23,18,21,20,24,19,22,23,18,21,20],
    "Study_Hours": [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,2,2.5,3,3.5,4,4.5,5,5.5,6,1,1.5,2.5,3,3.5,4,4.5,5,5.5,6,2,3,4,5,6,1.5,2.5,3.5,4.5,5.5],
    "Attendance": [55,60,65,70,75,80,85,90,95,98,92,62,68,72,78,82,88,94,96,99,58,64,69,74,79,84,89,93,97,91,57,67,77,87,97,61,71,81,91,96],
    "Previous_Score": [35,42,48,54,60,66,72,78,84,90,96,40,46,52,58,64,70,76,82,88,38,44,50,56,62,68,74,80,86,92,36,48,60,72,84,42,54,66,78,90],
    "Internet": ["No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes"],
    "Study_Method": ["Self","Group","Online","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self","Online","Group","Self"]
}

df = pd.DataFrame(data)

# 2. Create target variable
df["Result"] = np.where(
    (df["Study_Hours"].ge(3.5).astype(int) +
     df["Attendance"].ge(75).astype(int) +
     df["Previous_Score"].ge(60).astype(int)) >= 2,
    "Pass", "Fail"
)

# 3. Introduce missing values for preprocessing
df.loc[[7,30], "Study_Hours"] = np.nan
df.loc[[16], "Attendance"] = np.nan
df.loc[[21], "Previous_Score"] = np.nan
df.loc[[11,27], "Internet"] = np.nan
df.loc[[34], "Study_Method"] = np.nan

# 4. Save dataset as CSV
df.to_csv("decision_tree_dataset.csv", index=False)
print("Dataset:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# 5. Separate features and target
X = df.drop(columns=["Student_ID", "Result"])
y = df["Result"]

# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 7. Define numerical and categorical features
numeric = ["Age", "Study_Hours", "Attendance", "Previous_Score"]
categorical = ["Internet", "Study_Method"]

# 8. Define different feature subsets
subsets = {
    "Academic": ["Study_Hours", "Attendance", "Previous_Score"],
    "Demographic/access": ["Age", "Internet", "Study_Method"],
    "All features": X.columns.tolist()
}

# 9. Train and evaluate Decision Tree models
results, models = [], {}

for name, cols in subsets.items():
    nums = [c for c in cols if c in numeric]
    cats = [c for c in cols if c in categorical]

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cats)
    ])

    model = Pipeline([
        ("preprocessing", prep),
        ("classifier", DecisionTreeClassifier(
            criterion="gini", max_depth=4, random_state=42
        ))
    ])

    model.fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])

    results.append({
        "Feature Subset": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label="Pass", zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label="Pass", zero_division=0),
        "F1-score": f1_score(y_test, pred, pos_label="Pass", zero_division=0)
    })

    models[name] = model

# 10. Display performance comparison
results_df = pd.DataFrame(results).round(3)
print("\nPerformance Comparison:\n", results_df.to_string(index=False))

# 11. Plot accuracy comparison
plt.figure(figsize=(8, 4))
plt.bar(results_df["Feature Subset"], results_df["Accuracy"])
plt.ylim(0, 1.05)
plt.xlabel("Feature Subset")
plt.ylabel("Accuracy")
plt.title("Decision Tree Performance Comparison")
plt.tight_layout()
plt.show()

# 12. Visualize Decision Tree using all features
best_model = models["All features"]
feature_names = best_model.named_steps["preprocessing"].get_feature_names_out()

plt.figure(figsize=(18, 8))
plot_tree(
    best_model.named_steps["classifier"],
    feature_names=feature_names,
    class_names=best_model.named_steps["classifier"].classes_,
    filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree Using All Features")
plt.tight_layout()
plt.show()