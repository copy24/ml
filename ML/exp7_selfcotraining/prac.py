import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Create dataset manually using a dictionary
data = {
    "study_hours": [1,2,1.5,3,2.5,4,1,3.5,2,4.5,1.5,2.5,3,5,2,3.5,4,1,5.5,2.5,3,4.5,1.5,5,5,6,7,4.5,8,6.5,5.5,7.5,4,6,8,5,7,4.5,6.5,8.5,5.5,7,4,6,7.5,5,8,6.5],
    "attendance": [48,55,60,65,58,70,45,62,67,72,52,64,69,75,50,68,73,47,78,59,66,71,56,74,78,85,92,75,98,88,82,95,72,86,96,80,90,76,89,100,83,91,74,87,94,79,97,85],
    "previous_score": [28,35,40,45,38,50,25,42,47,53,32,44,49,56,30,46,52,27,60,39,43,54,34,57,60,68,82,58,90,76,65,85,55,72,88,63,79,57,74,95,66,81,54,70,86,61,92,75],
    "study_method": ["Self","Self","Online","Self","Group","Online","Self","Group","Self","Group","Online","Self","Group","Online","Self","Group","Online","Self","Group","Online","Self","Group","Online","Group","Group","Online","Group","Self","Group","Online","Group","Online","Self","Group","Online","Group","Online","Self","Group","Online","Group","Online","Self","Group","Online","Group","Online","Group"],
    "internet_access": ["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes"],
    "assignment_status": ["Pending","Pending","Pending","Complete","Pending","Complete","Pending","Complete","Pending","Complete","Pending","Complete","Pending","Complete","Pending","Complete","Complete","Pending","Complete","Pending","Complete","Complete","Pending","Complete","Complete","Complete","Complete","Pending","Complete","Complete","Complete","Complete","Pending","Complete","Complete","Complete","Complete","Pending","Complete","Complete","Complete","Complete","Pending","Complete","Complete","Pending","Complete","Complete"],
    "result": ["Fail"]*24 + ["Pass"]*24
}

df = pd.DataFrame(data)
df.loc[[2,12], "study_hours"] = np.nan
df.loc[[8], "attendance"] = np.nan
df.loc[[14], "study_method"] = np.nan
df.loc[[20], "internet_access"] = np.nan
df.loc[[30], "assignment_status"] = np.nan
# df.to_csv("student_performance.csv", index=False)
df = pd.read_csv("student_performance.csv")
print("Dataset saved as student_performance.csv")
print("Dataset shape:", df.shape)

# 2. Separate features and target
X = df.drop(columns="result")
y = df["result"].map({"Fail": 0, "Pass": 1})
num_cols = ["study_hours", "attendance", "previous_score"]
cat_cols = ["study_method", "internet_access", "assignment_status"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
indices = np.arange(len(X_train))
label_idx, unlabel_idx = train_test_split(indices, test_size=0.70, random_state=42, stratify=y_train)
y_semi = np.full(len(y_train), -1, dtype=int)
y_semi[label_idx] = y_train.iloc[label_idx].to_numpy()
print(f"Initially labelled: {len(label_idx)} | Unlabelled: {len(unlabel_idx)}")

# 4. Preprocess numerical and categorical data
num_prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_prep = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
preprocessor = ColumnTransformer([("num", num_prep, num_cols), ("cat", cat_prep, cat_cols)])
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
base_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)

# 5. Evaluation function
results = []
def evaluate(name, actual, predicted):
    scores = [accuracy_score(actual, predicted), precision_score(actual, predicted, zero_division=0), recall_score(actual, predicted, zero_division=0), f1_score(actual, predicted, zero_division=0)]
    results.append([name] + scores)
    print(f"{name}: Accuracy={scores[0]:.3f}, Precision={scores[1]:.3f}, Recall={scores[2]:.3f}, F1={scores[3]:.3f}")

# 6. Train initial model using labelled data only
initial_model = clone(base_model).fit(X_train_p[label_idx], y_train.iloc[label_idx])
evaluate("Initial model", y_test, initial_model.predict(X_test_p))

# 7. SELF-TRAINING
self_model = SelfTrainingClassifier(estimator=clone(base_model), threshold=0.80, max_iter=10, verbose=True)
self_model.fit(X_train_p, y_semi)
print("Self-Training pseudo-labels added:", int((self_model.labeled_iter_ > 0).sum()))
print("Self-Training iterations:", self_model.n_iter_)
evaluate("Self-Training", y_test, self_model.predict(X_test_p))

# 8. CO-TRAINING: two different feature views
view1, view2 = num_cols, cat_cols
prep1 = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), view1)])
prep2 = ColumnTransformer([("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), view2)])
A, A_test = prep1.fit_transform(X_train), prep1.transform(X_test)
B, B_test = prep2.fit_transform(X_train), prep2.transform(X_test)
model1 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
model2 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=43)
L1, L2 = list(label_idx), list(label_idx)
Y1 = {i: int(y_train.iloc[i]) for i in label_idx}
Y2 = Y1.copy()
U1, U2 = set(unlabel_idx), set(unlabel_idx)

for iteration in range(10):
    m1 = clone(model1).fit(A[L1], [Y1[i] for i in L1])
    m2 = clone(model2).fit(B[L2], [Y2[i] for i in L2])
    add1, add2 = [], []
    if U1:
        order = sorted(U1)
        prob = m2.predict_proba(B[order])
        add1 = [(i, int(c)) for i,c,p in zip(order, prob.argmax(axis=1), prob.max(axis=1)) if p >= 0.80]
    if U2:
        order = sorted(U2)
        prob = m1.predict_proba(A[order])
        add2 = [(i, int(c)) for i,c,p in zip(order, prob.argmax(axis=1), prob.max(axis=1)) if p >= 0.80]
    if not add1 and not add2:
        break
    for i,c in add1:
        L1.append(i); Y1[i] = c; U1.remove(i)
    for i,c in add2:
        L2.append(i); Y2[i] = c; U2.remove(i)
    print(f"Co-Training iteration {iteration+1}: added {len(add1)} to view 1 and {len(add2)} to view 2")

m1 = clone(model1).fit(A[L1], [Y1[i] for i in L1])
m2 = clone(model2).fit(B[L2], [Y2[i] for i in L2])
co_pred = ((m1.predict_proba(A_test) + m2.predict_proba(B_test)) / 2).argmax(axis=1)
print("Co-Training pseudo-labels: view 1 =", len(L1)-len(label_idx), ", view 2 =", len(L2)-len(label_idx))
evaluate("Co-Training", y_test, co_pred)

# 9. Compare performance and plot graph
comparison = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]).set_index("Model")
print("\nPerformance comparison:\n", comparison.round(3))
comparison[["Accuracy", "F1-score"]].plot(kind="bar", ylim=(0,1), rot=0, figsize=(8,4))
plt.title("Performance Before and After Label Propagation")
plt.ylabel("Score")
plt.tight_layout()
plt.show()