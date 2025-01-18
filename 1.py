from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 设置环境变量以禁用 oneDNN 优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 载入数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建模型
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))  # 输入层修改为 8 个特征
model.add(Dense(32, activation='relu'))  # 第二个隐藏层
model.add(Dense(1, activation='linear'))  # 输出层
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# 绘制损失图
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss', linestyle='-', color='b')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='r')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.grid(True)
plt.show()

# 输出模型的性能评估
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Training Loss: {train_loss}")
print(f"Test Loss: {test_loss}")
