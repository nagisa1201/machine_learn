import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV  # 使用 LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 载入数据
data_path = r"F:\broserload\mushroom\agaricus-lepiota.data"
data = pd.read_csv(data_path, header=None)  # 读取没有表头的数据

# 查看数据的一部分
print(data.head())

# 将所有的类别型数据转换为数值型
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# 查看编码后的数据
print(data_encoded.head())

# 使用 SimpleImputer 填补缺失值（使用均值填补）
imputer = SimpleImputer(strategy='mean')  # 也可以选择 'median' 或 'most_frequent'
data_imputed = pd.DataFrame(imputer.fit_transform(data_encoded), columns=data_encoded.columns)

# 查看填补后的数据
print(data_imputed.head())

# 假设最后一列是目标变量，其他列是特征
X = data_imputed.drop(columns=[data_imputed.columns[-1]])  # 删除最后一列
y = data_imputed[data_imputed.columns[-1]]  # 最后一列作为目标

# 加入高斯噪声
noise = np.random.normal(0, 1, X.shape)
X_noisy = X + noise

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 LogisticRegressionCV 模型
model = LogisticRegressionCV(max_iter=1000)  # 增加 max_iter 参数，避免收敛问题
model.fit(X_train, y_train)

# 绘制损失图
plt.plot(range(1, len(model.scores_[1]) + 1), model.scores_[1], label='Training Loss')  # 使用 `model.scores_`
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()
