import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading Dataset
df = pd.read_csv("data\heart_failure_clinical_records_dataset.csv")

#Basic checks
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

print("\nMissing values per column:\n", df.isnull().sum())

print("\nTarget distribution (DEATH_EVENT):\n", df["DEATH_EVENT"].value_counts())
print("\nTarget distribution (%):\n", (df["DEATH_EVENT"].value_counts(normalize=True) * 100).round(2))

print("\nDtypes:\n", df.dtypes)

# Separate features and target
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain death rate:", round(y_train.mean(), 3))
print("Test death rate:", round(y_test.mean(), 3))

# Dividing features 
binary_features = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking"
]

continuous_features = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]

# Make safe copies
X_train = X_train.copy()
X_test = X_test.copy()

# Log transform skewed features
skewed_features = [
    "creatinine_phosphokinase",
    "serum_creatinine"
]

for col in skewed_features:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

# Create scaled versions for LR and NN
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
