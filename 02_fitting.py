import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

'''
1. 生成数据
2.划分训练集和测试集
3.定义模型
4.训练模型
5.预测结果，计算误差
6.评估模型
'''

# 1. 生成数据
X= np.linspace(-3,3,300).reshape(-1,1)
y = np.sin(X) + np.random.uniform(-0.5,0.5,size=X.shape).reshape(-1,1)

# 画出散点图
fig , ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,y,c='y')
ax[1].scatter(X,y,c='y')
ax[2].scatter(X,y,c='y')
# plt.show()

# 2.划分训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.定义模型
model = LinearRegression()

########################################################################
#           1次多项式拟合
########################################################################
# 一、欠拟合（直线）
x_train1 = train_X
x_test1 = test_X

# 4.训练模型
model.fit(x_train1, train_y)

# 打印模型参数
print(model.coef_)
print(model.intercept_)

# 5.预测结果，计算误差
y_pred1 = model.predict(x_test1)
test_loss1 = mean_squared_error(test_y, y_pred1)
train_loss1 = mean_squared_error(train_y, model.predict(x_train1))

# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(x_test1, y_pred1, c='r')
ax[0].set_title(f'Train Error: {mean_squared_error(train_y, model.predict(x_train1)):.4f}\nTest Error: {mean_squared_error(test_y, y_pred1):.4f}')
# plt.show()

########################################################################
#           5次多项式拟合
########################################################################
# 二、欠拟合（5次多项式）
poly5 = PolynomialFeatures(degree=5) 
x_train2 = poly5.fit_transform(train_X)
x_test2 = poly5.transform(test_X)

# 4.训练模型
model.fit(x_train2, train_y)

# 打印模型参数
print(model.coef_)
print(model.intercept_)

# 5.预测结果，计算误差
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(test_y, y_pred2)
train_loss2 = mean_squared_error(train_y, model.predict(x_train2))

# 画出拟合曲线，并写出训练误差和测试误差
ax[1].plot(X, model.predict(poly5.transform(X)), c='r')
ax[1].set_title(f'Train Error: {mean_squared_error(train_y, model.predict(x_train2)):.4f}\nTest Error: {mean_squared_error(test_y, y_pred2):.4f}')
# plt.show()

############################################################
#           20次多项式拟合（过拟合）
############################################################
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(train_X)
x_test3 = poly20.transform(test_X)

# 4.训练模型
model.fit(x_train3, train_y)

# 打印模型参数
print(model.coef_)
print(model.intercept_)

# 5.预测结果，计算误差
y_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(test_y, y_pred3)
train_loss3 = mean_squared_error(train_y, model.predict(x_train3))

# 画出拟合曲线，并写出训练误差和测试误差
ax[2].plot(X, model.predict(poly20.transform(X)), c='r')
ax[2].set_title(f'Train Error: {mean_squared_error(train_y, model.predict(x_train3)):.4f}\nTest Error: {mean_squared_error(test_y, y_pred1):.4f}')
plt.show()
