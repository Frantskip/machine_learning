import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv(r"D:\桌面\machine_learning\09_advertising\Advertising.csv")

# 1.读取数据
dataset.dropna(inplace=True)
dataset.drop(columns=dataset.columns[0], inplace=True)

dataset.info()
print(dataset.head())

# 2.划分数据集
X = dataset.drop(columns=['Sales'])
y = dataset['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3.标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.创建模型
# 4.1 线性回归模型
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
print("model_lr.coef_:", model_lr.coef_)
print("model_lr.intercept_:", model_lr.intercept_)

# 4.2 SGDRegressor模型
model_sgd = SGDRegressor()
model_sgd.fit(X_train, y_train)
print("model_sgd.coef_:", model_sgd.coef_)
print("model_sgd.intercept_:", model_sgd.intercept_)

# 5. 预测
y_pred_lr = model_lr.predict(X_test)
y_pred_sgd = model_sgd.predict(X_test)

# 6. 使用均方误差评估模型
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
print("mse_lr:", mse_lr)
print("mse_sgd:", mse_sgd)
