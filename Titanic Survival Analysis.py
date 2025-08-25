# ========================
# 📌 Titanic Survival Analysis using Seaborn
# ========================

# Step 1: Setup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
titanic = sns.load_dataset("titanic")
print(titanic.head())
print(titanic.info())


# ========================
# Step 2: Survival Rate by Category
# ========================

# Count survival
sns.countplot(x="survived", data=titanic)
plt.title("Survival Counts")
plt.show()

# Survival by gender
sns.barplot(x="sex", y="survived", data=titanic)
plt.title("Survival Rate by Gender")
plt.show()

# Survival by passenger class
sns.barplot(x="class", y="survived", data=titanic)
plt.title("Survival Rate by Passenger Class")
plt.show()


# ========================
# Step 3: Age Distribution
# ========================

# Age distribution histogram
sns.histplot(titanic["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# Age vs survival (boxplot)
sns.boxplot(x="survived", y="age", data=titanic)
plt.title("Age vs Survival")
plt.show()


# ========================
# Step 4: Survival by Multiple Factors
# ========================

# Survival by gender & class
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
plt.title("Survival by Gender & Class")
plt.show()

# Survival by embarkation port & gender
sns.catplot(x="embarked", y="survived", hue="sex", kind="bar", data=titanic)
plt.title("Survival by Embarkation Port & Gender")
plt.show()


# ========================
# Step 5: Correlation Heatmap
# ========================

# Compute correlation matrix
corr = titanic.corr(numeric_only=True)
print(corr)

# Heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ========================
# Step 6: Pairplot of Key Features
# ========================

sns.pairplot(titanic[["age", "fare", "survived", "pclass"]].dropna(), hue="survived")
plt.show()
