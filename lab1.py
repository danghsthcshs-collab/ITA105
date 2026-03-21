import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Bài 1: Tạo dữ liệu
# =========================
data = {
    "Product": ["A", "B", "C", "D", "E"],
    "Price": [100, 200, -50, 150, 300],
    "StockQuantity": [10, -5, 20, 15, 30],
    "Category": ["Electronics", None, "Clothing", None, "Electronics"]
}

df = pd.DataFrame(data)

print("Dữ liệu ban đầu:")
print(df)

# =========================
# Bài 2: Xử lý dữ liệu thiếu
# =========================

# Điền giá trị thiếu bằng mode
df["Category"].fillna(df["Category"].mode()[0], inplace=True)

print("\nSau khi xử lý dữ liệu thiếu:")
print(df)

# So sánh với dropna
df_drop = df.dropna()
print("\nKích thước sau khi dropna:", df_drop.shape)

# =========================
# Bài 3: Xử lý dữ liệu lỗi
# =========================

# Price phải >= 0
df = df[df["Price"] >= 0]

# StockQuantity >= 0
df = df[df["StockQuantity"] >= 0]

print("\nSau khi xử lý dữ liệu lỗi:")
print(df)

# =========================
# Bài 4: Vẽ biểu đồ
# =========================

plt.bar(df["Product"], df["Price"])
plt.title("Price of Products")
plt.xlabel("Product")
plt.ylabel("Price")
plt.show()