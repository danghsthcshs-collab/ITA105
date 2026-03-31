import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. Tạo dataset (Dựa trên bảng Hours/Score trong ảnh)
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [2, 3.5, 5, 5.5, 7, 7.5, 8, 9, 9.2, 9.5]
}
df = pd.DataFrame(data)

# 2. Tách dữ liệu Input (X) và Output (y)
X = df[["Hours"]]
y = df["Score"]

# 3. Chia tập dữ liệu (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo mô hình
model = LinearRegression()

# 5. Huấn luyện mô hình (Train)
model.fit(X_train, y_train)

# 6. Dự đoán trên tập dữ liệu mới
y_pred = model.predict(X_test)

# 7. Dự đoán giá trị cụ thể (Ví dụ: học 6 giờ)
new_hour = np.array([[6]])
print(f"Học 6 giờ dự kiến được: {model.predict(new_hour)[0]:.2f} điểm")

# 8. Đánh giá mô hình (R2 Score)
score = r2_score(y_test, y_pred)
print(f"R2 Score: {score:.4f}")

# 9. Vẽ biểu đồ trực quan hóa kết quả
plt.scatter(X, y, color='blue', label='Thực tế')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Đường dự đoán')
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.title("Linear Regression: Hours vs Score")
plt.legend()
plt.show()