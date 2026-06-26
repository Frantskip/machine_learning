from sklearn.linear_model import LinearRegression
import numpy as np

# 定义损失函数
def J(beta):
    return np.sum((X@beta - y)**2) / n

# 定义梯度函数
def gradient(beta):
    return 2/n * X.T @ (X@beta - y)

# 1.定义数据
X = np.array([[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]])
y = np.array([[55],[65],[70],[75],[85],[50],[60],[72],[80],[58]])

n = X.shape[0]

# 2.数据处理
X = np.hstack((np.ones((n, 1)), X))

# 3.初始化参数以及超参数
alpha = 0.01
iter = 10000
beta = np.array([[1],[1]])

# 定义列表 保存每次迭代的参数值
beta0 = []
beta1 = []

# 重复迭代
# for i in range(iter):
while (np.abs(grad := gradient(beta)) > 1e-10).any() and (iter := iter - 1) > 0:
    beta0.append(beta[0,0])
    beta1.append(beta[1,0])
    # 5.更新参数
    beta = beta - alpha * grad
    
    # 6.打印损失函数值
    if iter % 1000 == 0:
        print(f"beta: {beta.reshape(-1)}\tJ(beta): {J(beta)}")

print(f"iter: {iter}")

# 4. 可视化参数迭代过程
import matplotlib.pyplot as plt

# plt.plot(beta0, label='beta0')
plt.plot(beta0,beta1 )
plt.show()