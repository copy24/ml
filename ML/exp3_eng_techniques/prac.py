import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. CREATE DATASET USING DICTIONARY
data = {
    'study_hours':[6.9,2.1,4.9,np.nan,2.5,3.7,4.0,5.3,6.2,1.1,2.8,5.2,1.6,8.0,6.8,1.3,5.0,np.nan,1.0,2.3,2.2,4.2,5.0,4.2,7.4,6.7,3.8,2.4,3.5,7.0,3.4,7.9,5.0,2.7,np.nan,5.6,4.6,3.0,5.4,4.4,5.6,4.8,1.8,7.4,1.1,3.2,2.2,3.4],
    'attendance':[82,78,86,75,92,69,78,76,np.nan,63,56,64,52,66,70,83,92,72,82,77,63,62,74,90,80,86,87,82,64,np.nan,77,53,66,69,52,79,78,56,64,95,73,65,93,74,82,95,78,60],
    'assignment_score':[83,96,66,80,93,np.nan,87,69,74,96,89,50,66,81,89,51,83,72,47,72,68,68,np.nan,63,86,62,92,83,86,65,52,43,66,69,64,48,45,93,65,92,90,np.nan,92,97,79,42,79,71],
    'aptitude_score':[86,58,80,59,73,64,49,56,39,76,93,np.nan,46,70,52,60,35,97,69,72,41,44,92,90,79,80,96,79,74,45,50,49,38,62,38,94,75,np.nan,52,83,93,100,40,64,98,73,87,60],
    'sleep_hours':[8.4,8.8,5.3,5.7,6.1,8.8,4.3,7.0,6.9,7.8,4.4,4.8,8.0,8.7,np.nan,6.6,7.6,5.0,8.1,4.1,7.1,8.2,4.6,5.8,5.4,7.9,4.9,6.0,6.9,4.2,5.9,4.9,np.nan,4.9,6.0,4.2,5.2,5.1,7.4,7.4,6.0,4.5,6.6,6.6,6.6,5.9,5.3,4.1],
    'internet':['Yes','Yes','Yes','Yes','Yes','No','No','Yes','Yes',np.nan,'Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes',np.nan,'Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','No','Yes','Yes','Yes','No'],
    'study_method':['Online','Coaching','Coaching','Coaching','Coaching','Group',np.nan,'Online','Online','Self','Self','Group','Group','Online','Group','Group','Online','Coaching','Coaching','Online','Group','Coaching','Self','Self','Group','Online','Group','Group','Online','Online','Online','Online','Online','Online','Self','Group','Coaching','Coaching','Group',np.nan,'Self','Online','Coaching','Coaching','Self','Coaching','Online','Self'],
    'skill_level':['Advanced','Intermediate',np.nan,'Intermediate','Intermediate','Intermediate','Beginner','Intermediate','Beginner','Intermediate','Advanced','Intermediate','Beginner','Intermediate','Beginner','Intermediate','Beginner','Advanced','Intermediate','Intermediate','Beginner','Beginner','Advanced','Advanced','Advanced','Advanced','Advanced','Advanced','Intermediate','Beginner','Beginner',np.nan,'Beginner','Intermediate','Beginner','Advanced','Intermediate','Intermediate','Beginner','Advanced','Advanced','Advanced','Beginner','Intermediate','Advanced','Intermediate','Advanced','Intermediate'],
    'Result':['Pass','Pass','Pass','Pass','Pass','Fail','Pass','Pass','Fail','Fail','Pass','Fail','Fail','Pass','Pass','Fail','Pass','Pass','Fail','Fail','Fail','Fail','Pass','Pass','Pass','Pass','Pass','Pass','Pass','Fail','Fail','Pass','Fail','Fail','Fail','Pass','Pass','Fail','Pass','Pass','Pass','Pass','Fail','Pass','Pass','Pass','Pass','Fail']
}

df = pd.DataFrame(data)
# df.to_csv("feature_engineering_dataset.csv", index=False)

# To load the generated CSV instead, use:
# df = pd.read_csv("feature_engineering_dataset.csv")

print("Dataset:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# 2. SEPARATE FEATURES AND TARGET
X = df.drop(columns=['Result'])
le = LabelEncoder()
y = le.fit_transform(df['Result'])
print("\nTarget encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

num = ['study_hours','attendance','assignment_score','aptitude_score','sleep_hours']
cat = ['internet','study_method']
ordinal = ['skill_level']

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. PREPROCESSING FUNCTION
def prep(nums, scaler=None):
    nsteps = [('impute', SimpleImputer(strategy='median'))]
    if scaler is not None:
        nsteps.append(('scale', scaler))

    return ColumnTransformer([
        ('num', Pipeline(nsteps), nums),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat),
        ('ord', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(
                categories=[['Beginner','Intermediate','Advanced']],
                handle_unknown='use_encoded_value', unknown_value=-1
            ))
        ]), ordinal)
    ], sparse_threshold=0)

# 5. MODEL EVALUATION FUNCTION
results = []

def evaluate(name, xtr, xte, nums, scaler=None, extra=None):
    steps = [('prep', prep(nums, scaler))] + (extra or [])
    steps.append(('model', LogisticRegression(max_iter=2000, random_state=42)))

    model = Pipeline(steps)
    model.fit(xtr, y_train)
    pred = model.predict(xte)

    results.append([
        name, accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0)
    ])

# 6. BASELINE MODEL
evaluate('Baseline', X_train, X_test, num)

# 7. STANDARDIZATION
evaluate('Standardization', X_train, X_test, num, StandardScaler())

# 8. NORMALIZATION
evaluate('Normalization', X_train, X_test, num, MinMaxScaler())

# 9. FEATURE CREATION
Xftr, Xfte = X_train.copy(), X_test.copy()
for d in [Xftr, Xfte]:
    d['engagement_score'] = (d.attendance + d.assignment_score) / 2

evaluate('Feature Creation', Xftr, Xfte, num + ['engagement_score'], StandardScaler())

# 10. POLYNOMIAL FEATURES
evaluate('Polynomial Features', X_train, X_test, num, StandardScaler(), [
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scale_all', StandardScaler())
])

# 11. PCA
evaluate('PCA 95%', X_train, X_test, num, StandardScaler(), [
    ('scale_all', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

# 12. DISPLAY COMPARISON
comparison = pd.DataFrame(results, columns=[
    'Technique','Accuracy','Precision','Recall','F1 Score'
])
print("\nMODEL PERFORMANCE COMPARISON:\n")
print(comparison.round(3).to_string(index=False))