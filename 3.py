import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 载入数据
data = pd.read_csv(r"F:/broserload/car+evaluation/car.data", header=None)

# 对每列类别特征进行编码
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':  # 仅对字符串列进行编码
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# 分割特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用核SVM
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# 测试集预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 绘制决策函数
decision_function = model.decision_function(X_test)

# 绘制决策函数（即分类边界的距离）
plt.figure()
plt.plot(decision_function, label="Decision Function")
plt.legend()
plt.title("Decision Function for Test Set")
plt.show()
