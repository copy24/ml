
import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create categorical dataset using dictionary
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Mild','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'Play': ['No','No','Yes','No','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
# df.to_csv('naive_bayes_categorical.csv', index=False)
print("Dataset:\n", df)
print("\nDataset Shape:", np.shape(df))

# 2. Separate input features and target
X = df.drop('Play', axis=1)
y = df['Play']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Encode categorical input and target data
encoder, le = OrdinalEncoder(), LabelEncoder()
X_train = encoder.fit_transform(X_train).astype(int)
X_test = encoder.transform(X_test).astype(int)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("\nFeature Encoding:")
for col, cats in zip(X.columns, encoder.categories_):
    print(col, dict(zip(cats, range(len(cats)))))

print("Target Encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# 5. Train the Categorical Naive Bayes model
model = CategoricalNB(alpha=1.0)
model.fit(X_train, y_train)

# 6. Predict test data
y_pred = model.predict(X_test)

# 7. Evaluate model performance
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# 8. Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes - Confusion Matrix')
plt.tight_layout()
plt.show()