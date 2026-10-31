# EXPERIMENT 01: Exploratory Data Analysis (EDA)

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 2. Create dataset manually
data = [
    [1,2,75,68,7,72,'Pass'], [2,1.5,60,55,6.5,58,'Pass'],
    [3,4,90,82,8,88,'Pass'], [4,.5,45,38,6,35,'Fail'],
    [5,3,80,74,7.5,79,'Pass'], [6,1,50,42,5.5,39,'Fail'],
    [7,5,95,90,8,94,'Pass'], [8,2.5,70,65,7,68,'Pass'],
    [9,1,55,48,6,45,'Pass'], [10,3.5,85,80,7.5,84,'Pass'],
    [11,.8,40,35,5,30,'Fail'], [12,2.2,72,67,7,70,'Pass'],
    [13,4.5,92,88,8.5,91,'Pass'], [14,1.2,58,50,6,48,'Pass'],
    [15,2.8,78,73,7,76,'Pass'], [16,.4,35,30,5,28,'Fail'],
    [17,3.2,88,79,7.5,82,'Pass'], [18,2,68,62,6.5,65,'Pass'],
    [19,6,98,95,9,97,'Pass'], [20,1.8,65,60,7,63,'Pass'],
    [21,2.7,76,71,7.5,74,'Pass'], [22,.7,42,36,5.5,33,'Fail'],
    [23,3.8,89,84,8,87,'Pass'], [24,1.6,62,57,6.5,59,'Pass'],
    [25,25,150,150,20,100,'Pass'], [26,2.4,None,69,7,71,'Pass'],
    [27,None,73,66,6.5,67,'Pass'], [28,3.1,82,None,7.5,80,'Pass'],
    [29,1.3,57,52,None,49,'Pass'], [30,2.9,79,75,7,None,'Pass'],
    [1,2,75,68,7,72,'Pass'], [3,4,90,82,8,88,'Pass']
]

cols = ['Student_ID','Study_Hours','Attendance','Previous_Score',
        'Sleep_Hours','Final_Score','Result']

df = pd.DataFrame(data, columns=cols)

# Generate CSV dataset
df.to_csv('student_performance_dataset.csv', index=False)

# 3. Explore dataset
print("\nFirst Five Records:\n", df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Description:\n", df.describe())
print("\nDataset Shape:", df.shape)

# 4. Identify and handle missing values
print("\nMissing Values:\n", df.isnull().sum())

num = ['Study_Hours','Attendance','Previous_Score',
       'Sleep_Hours','Final_Score']

df[num] = df[num].fillna(df[num].median())
df['Result'] = df['Result'].fillna(df['Result'].mode()[0])

# 5. Identify and remove duplicates
print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()

# 6. Identify and handle outliers using IQR
for col in num:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col} - Outliers: {len(outliers)}")
    df[col] = df[col].clip(lower, upper)

print("\nCleaned Dataset:\n", df)
print("\nFinal Shape:", df.shape)

# 7. Data visualization
sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 3, figsize=(15, 9))

sns.histplot(df['Final_Score'], kde=True, ax=ax[0,0])
ax[0,0].set_title("Histogram")

sns.boxplot(data=df[num], ax=ax[0,1])
ax[0,1].set_title("Boxplot")
ax[0,1].tick_params(axis='x', rotation=30)

sns.countplot(data=df, x='Result', ax=ax[0,2])
ax[0,2].set_title("Pass/Fail Count")

sns.scatterplot(data=df, x='Study_Hours', y='Final_Score',
                hue='Result', ax=ax[1,0])
ax[1,0].set_title("Study Hours vs Final Score")

sns.heatmap(df[num].corr(), annot=True, cmap='coolwarm',
            fmt='.2f', ax=ax[1,1])
ax[1,1].set_title("Correlation Heatmap")

ax[1,2].axis('off')
plt.tight_layout()
plt.show()

# 8. Model training and evaluation
X = df[['Study_Hours','Attendance','Previous_Score','Sleep_Hours']]
y = df['Result'].map({'Fail':0, 'Pass':1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. Evaluation metrics
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nExperiment Completed Successfully!")