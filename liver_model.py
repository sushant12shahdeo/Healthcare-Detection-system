import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("liver.csv")

# Rename target column to binary (0 = no disease, 1 = liver disease)
df['Liver_Disease'] = df['Liver_Disease'].apply(lambda x: 1 if x == 1 else 0)

X = df.drop("Liver_Disease", axis=1)
y = df["Liver_Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "liver_model.pkl")