import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =========================
# LOAD DATA (đổi tên file)
# =========================
df = pd.read_csv('data.csv', encoding='utf-16')

print("=== HEAD ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Lấy cột số
num_cols = df.select_dtypes(include=np.number).columns

# =========================
# HISTOGRAM + BOXPLOT
# =========================
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram - {col}")
    plt.show()

    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# =========================
# OUTLIERS (IQR)
# =========================
print("\n=== OUTLIERS ===")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")

# =========================
# MIN-MAX
# =========================
minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df[num_cols]), columns=num_cols)

# =========================
# Z-SCORE
# =========================
zscore = StandardScaler()
df_zscore = pd.DataFrame(zscore.fit_transform(df[num_cols]), columns=num_cols)

# =========================
# SO SÁNH PHÂN PHỐI
# =========================
for col in num_cols:
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title("Original")

    plt.subplot(1,3,2)
    sns.histplot(df_minmax[col], kde=True)
    plt.title("Min-Max")

    plt.subplot(1,3,3)
    sns.histplot(df_zscore[col], kde=True)
    plt.title("Z-Score")

    plt.suptitle(col)
    plt.show()

# =========================
# SCATTER (BÀI 3)
# =========================
if len(num_cols) >= 2:
    x = num_cols[0]
    y = num_cols[1]

    plt.scatter(df[x], df[y])
    plt.title("Before Scaling")
    plt.show()

    plt.scatter(df_minmax[x], df_minmax[y])
    plt.title("Min-Max")
    plt.show()

    plt.scatter(df_zscore[x], df_zscore[y])
    plt.title("Z-Score")
    plt.show()