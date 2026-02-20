import pandas as pd

# Load dataset
df =  pd.read_csv("C:\Users\manan\Downloads\healthcare-dataset-stroke-data.csv")

# ---- STEP 1: Basic Structure ----
print("===== BASIC STRUCTURE =====")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n")
print(df.head())
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n")
print(df.dtypes)

# ---- STEP 2: Missing Values ----
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ---- STEP 3: Class Distribution ----
print("\n===== CLASS DISTRIBUTION =====")
print("Counts:\n", df["stroke"].value_counts())
print("\nPercentages:\n", df["stroke"].value_counts(normalize=True))

# ---- STEP 4: Statistical Summary ----
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
