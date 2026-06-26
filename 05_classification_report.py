from sklearn.datasets import make_classification  # 生成分类数据集
from sklearn.metrics import classification_report # 分类报告
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
# 1.生成一个二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# print(X.shape)
# print(y.shape)
# 2.划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3.训练一个分类模型（这里以逻辑回归为例）
model = LogisticRegression()

# 4.训练模型
model.fit(X_train, y_train)

# 5.预测
y_pred = model.predict(X_test)

# 6.生成分类报告
report = classification_report(y_test, y_pred, digits=4)
print(report)

# 获取预测正类的概率值
y_pred_proba = model.predict_proba(X_test)[:, 1]
# print(y_pred_proba)

# 计算AUC值
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

# 画ROC曲线
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
# ---------------- 补充完整的ROC曲线绘制代码 ----------------
# 1. 计算ROC曲线的假正率(FPR)、真正率(TPR)和阈值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# 2. 创建绘图画布，设置尺寸
plt.figure(figsize=(8, 6))

# 3. 绘制ROC曲线，标注AUC值
plt.plot(fpr, tpr, color='#1f77b4', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')

# 4. 绘制随机猜测的对角线（作为参考基线）
plt.plot([0, 1], [0, 1], color='#d62728', linestyle='--', linewidth=1.5, label='Random Guess')

# 5. 设置图表标题、坐标轴标签
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)

# 6. 添加图例、显示网格
plt.legend(fontsize=11)
plt.grid(alpha=0.3, linestyle=':')

# 7. 显示图表
plt.show()