import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Rows, Columns:", df.shape)
print(df.head())
print(df['Churn'].value_counts())
print("\n--- Cleaning ---")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Before Rows:", df.shape)
df = df.dropna()
print("After Rows:", df.shape)
df = df.drop('customerID', axis=1)
print("Cleaning Done!")
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Graph ---")
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.show()
plt.figure(figsize=(7,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract vs Churn")
plt.show()
print("\n--- Model Training ---")
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)
print("\n--- Results ---")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))