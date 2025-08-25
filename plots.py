# ========================
# 📌 Seaborn Learning Roadmap (Full Script)
# ========================

# Step 1: Setup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
print("Available datasets:", sns.get_dataset_names())
df = sns.load_dataset("tips")
print(df.head())


# ========================
# Step 2: Basic Plots
# ========================

# Scatter Plot
sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()

# Line Plot
sns.lineplot(x="size", y="tip", data=df)
plt.show()

# Bar Plot
sns.barplot(x="day", y="total_bill", data=df)
plt.show()


# ========================
# Step 3: Categorical Plots
# ========================

# Count Plot
sns.countplot(x="day", data=df)
plt.show()

# Box Plot
sns.boxplot(x="day", y="total_bill", data=df)
plt.show()

# Violin Plot
sns.violinplot(x="day", y="total_bill", data=df)
plt.show()


# ========================
# Step 4: Distribution Plots
# ========================

# Histogram
sns.histplot(df["total_bill"], bins=20, kde=True)
plt.show()

# Kernel Density Estimate (KDE)
sns.kdeplot(df["tip"], fill=True)
plt.show()


# ========================
# Step 5: Relationship Plots
# ========================

# Pair Plot
sns.pairplot(df, hue="sex")
plt.show()

# Joint Plot
sns.jointplot(x="total_bill", y="tip", data=df, kind="hex")
plt.show()


# ========================
# Step 6: Heatmaps & Correlation
# ========================

# Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()


# ========================
# Step 7: Styling & Themes
# ========================

# Set style and palette
sns.set_style("whitegrid")
sns.set_palette("pastel")

sns.boxplot(x="day", y="total_bill", data=df)
plt.show()


# ========================
# Step 8: Regression Plots
# ========================

# Simple regression plot
sns.regplot(x="total_bill", y="tip", data=df)
plt.show()

# Regression without line (scatter only)
sns.regplot(x="total_bill", y="tip", data=df, fit_reg=False)
plt.show()

# Polynomial regression (quadratic fit)
sns.regplot(x="total_bill", y="tip", data=df, order=2)
plt.show()

# Regression with jitter for categorical-like x variable
sns.regplot(x="size", y="tip", data=df, x_jitter=0.1)
plt.show()

# Regression with multiple categories using lmplot (hue)
sns.lmplot(x="total_bill", y="tip", hue="sex", data=df, height=5, aspect=1.2)
plt.show()

# Faceted regression plots by day
sns.lmplot(x="total_bill", y="tip", col="day", hue="sex", data=df, height=4, aspect=1)
plt.show()

