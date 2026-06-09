import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_titanic.csv")

# -------------------------
# Basic Dataset Information
# -------------------------
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------
# Visualization 1
# Survival Distribution
# -------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.savefig("eda_survival_distribution.png")
plt.show()

# -------------------------
# Visualization 2
# Age Distribution
# -------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.savefig("eda_age_distribution.png")
plt.show()

# -------------------------
# Visualization 3
# Passenger Class Distribution
# -------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.savefig("eda_pclass_distribution.png")
plt.show()

# -------------------------
# Correlation Heatmap
# -------------------------
df_corr = df.copy()

df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

plt.figure(figsize=(8,6))
sns.heatmap(
    df_corr.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.savefig("eda_correlation_heatmap.png")
plt.show()

print("\nEDA Project Completed Successfully!")