import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 读取数据集
data = pd.read_csv('F:/broserload/mushroom/agaricus-lepiota.data', header=None)

# 查看数据的前几行，了解数据结构
print(data.head())

# 给数据集指定正确的列名
data.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'Income']

# 选择特征和目标变量
X = data[['A', 'B', 'C']]  # 选择合适的特征列（根据你的数据修改）
y = data['Income']  # 目标变量

# 对所有的非数值列进行映射
# 在这个例子中假设所有列都是分类数据，需要转换
for col in X.columns:
    X[col] = X[col].map({'p': 0, 'e': 1, 'x': 2, 's': 3, 'b': 4, 'n': 5, 'f': 6, 't': 7, 'y': 8, 'w': 9, 'k': 10, 'l': 11, 'c': 12, 'g': 13, 'h': 14, 'a': 15})

# 对数据进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 创建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))
