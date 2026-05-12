import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("diabetes.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

replace_columns = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in replace_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].mean())

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===================================")
print("LOGISTIC REGRESSION MODEL")
print("===================================\n")

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Accuracy :", round(lr_accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, lr_pred, labels=[0,1]))

print("\n===================================")
print("RANDOM FOREST MODEL")
print("===================================\n")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Accuracy :", round(rf_accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred, labels=[0,1]))

print("\n===================================")
print("SUPPORT VECTOR MACHINE MODEL")
print("===================================\n")

svm_model = SVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("Accuracy :", round(svm_accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, svm_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, svm_pred, labels=[0,1]))

models = [
    "Logistic Regression",
    "Random Forest",
    "SVM"
]

accuracies = [
    lr_accuracy * 100,
    rf_accuracy * 100,
    svm_accuracy * 100
]

print("\n===================================")
print("MODEL COMPARISON")
print("===================================\n")

for model, acc in zip(models, accuracies):
    print(model, ":", round(acc, 2), "%")

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.show()

print("\n===================================")
print("CUSTOM PREDICTION")
print("===================================\n")

pregnancies = float(input("Enter Pregnancies: "))
glucose = float(input("Enter Glucose: "))
bp = float(input("Enter Blood Pressure: "))
skin = float(input("Enter Skin Thickness: "))
insulin = float(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter Diabetes Pedigree Function: "))
age = float(input("Enter Age: "))

custom_data = pd.DataFrame([[
    pregnancies,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]], columns=X.columns)

custom_scaled = scaler.transform(custom_data)

prediction = rf_model.predict(custom_scaled)

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:
    print("Person May Have Diabetes")
else:
    print("Person Does Not Have Diabetes")

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
