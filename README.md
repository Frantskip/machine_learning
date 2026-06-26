# Machine Learning — 从基础到实践

一个系统性的机器学习学习项目，涵盖从数学基础、经典算法原理到实际应用的完整流程。所有代码基于 **Python** 生态（NumPy、Scikit-learn、Matplotlib、Pandas），以 Jupyter Notebook 和 Python 脚本两种形式组织，适合逐主题学习和复现。

---

## 目录

- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [内容概览](#内容概览)
  - [01 — NumPy 线性代数基础](#01--numpy-线性代数基础)
  - [02 — 多项式拟合：欠拟合与过拟合](#02--多项式拟合欠拟合与过拟合)
  - [03 — 正则化：L1 (Lasso) 与 L2 (Ridge)](#03--正则化l1-lasso-与-l2-ridge)
  - [04 — 梯度下降法](#04--梯度下降法)
  - [05 — 分类模型评估指标](#05--分类模型评估指标)
  - [06 — K-近邻 (KNN)](#06--k-近邻-knn)
  - [07 — 心脏病预测：KNN + 网格搜索](#07--心脏病预测knn--网格搜索)
  - [08 — 线性回归](#08--线性回归)
  - [09 — 广告销量预测：多元线性回归](#09--广告销量预测多元线性回归)
  - [10 — 逻辑回归](#10--逻辑回归)
  - [12 — 感知机与逻辑门](#12--感知机与逻辑门)
  - [13 — 信息熵](#13--信息熵)
- [参考资料](#参考资料)

---

## 项目结构

```
machine_learning/
├── 01.ipynb                          # NumPy 向量与矩阵运算
├── 02_fitting.py                     # 多项式拟合（欠拟合 → 过拟合）
├── 03_regularization.py              # L1/L2 正则化对比
├── 04_gradient_desent1.ipynb         # 梯度下降：单变量二次函数
├── 04_gradient_desent2.ipynb         # 梯度下降：双谷函数
├── 05_classification_report.py       # 分类报告 + ROC/AUC
├── 05_classification_test.ipynb      # 混淆矩阵与分类指标
├── 06_knn_classification.ipynb       # KNN 分类器
├── 06_knn_regression.ipynb           # KNN 回归器
├── 06_knn_scaler.ipynb              # 特征缩放（归一化/标准化）
├── 07_heart_disease.py              # KNN + GridSearchCV 心脏病预测
├── 08_linear_regression.ipynb       # 线性回归（学习时长 vs 成绩）
├── 08_liner_regression_gradient.py  # 梯度下降手写线性回归
├── 08_sgd.ipynb                     # SGDRegressor 随机梯度下降
├── 09_advertising/
│   └── advertising.py               # 广告销量预测（LR vs SGD）
├── 10_logistic_regression/
│   ├── 1_api_test.ipynb             # LogisticRegression API 参数测试
│   ├── 3_digit_recognizer.py        # MNIST 手写数字识别
│   └── heart_disease.py             # 逻辑回归心脏病预测
├── 12_perceptron/
│   └── 1_logic_gate.py              # 感知机实现逻辑门
└── 13_supervise/
    └── 1_entropy.ipynb              # 信息熵可视化
```

---

## 环境依赖

| 包 | 用途 |
|---|---|
| `numpy` | 数值计算、矩阵运算 |
| `pandas` | 数据处理与 CSV 读取 |
| `matplotlib` | 数据可视化 |
| `seaborn` | 混淆矩阵热力图 |
| `scikit-learn` | 模型训练、评估、特征工程 |
| `joblib` | 模型持久化 |

安装命令：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 内容概览

### 01 — NumPy 线性代数基础

**文件:** `01.ipynb`

从零构建机器学习所需的线性代数直觉：

- 向量创建与属性（`shape`, `dtype`）
- 向量转置与 L2 范数（`np.linalg.norm`）
- 矩阵转置、矩阵加法、标量乘法
- 矩阵乘法 (`@` 运算符) 及其维度约束（`(n,k) @ (k,m) → (n,m)`）

> 关键认知：向量在 NumPy 中是一维数组，转置不改变其形状——这是初学者常见的困惑点。

---

### 02 — 多项式拟合：欠拟合与过拟合

**文件:** `02_fitting.py`

以 $y = \sin(x) + \epsilon$ 为真实分布，系统演示三种拟合状态：

| 多项式次数 | 现象 | 训练误差 | 测试误差 |
|---|---|---|---|
| 1 次 | **欠拟合 (Underfitting)** — 模型容量不足，无法捕获正弦曲线的非线性模式 | 高 | 高 |
| 5 次 | **适度拟合 (Good Fit)** — 较好逼近真实函数 | 低 | 低 |
| 20 次 | **过拟合 (Overfitting)** — 完美记忆训练点，但泛化能力极差 | 极低 | 极高 |

**技术要点：**
- `PolynomialFeatures` 进行特征构造，将一维输入映射到高维多项式空间
- 使用 `train_test_split` 划分训练集 (80%) 和测试集 (20%)
- 以均方误差 (MSE) 作为损失度量

---

### 03 — 正则化：L1 (Lasso) 与 L2 (Ridge)

**文件:** `03_regularization.py`

在 20 次多项式过拟合的基础上，对比引入正则化后的效果：

| 方法 | 正则化类型 | 惩罚项 | 效果 |
|---|---|---|---|
| 无正则化 | — | — | 系数幅值极大（>10¹³），严重过拟合 |
| **Lasso** (α=0.01) | L1 | $\lambda\sum\|\beta_i\|$ | 稀疏解：多数系数被压缩至**精确为零**，自动特征选择 |
| **Ridge** (α=1) | L2 | $\lambda\sum\beta_i^2$ | 系数整体收缩但非零，平滑且稳定 |

**可视化设计：**
- 上排三幅图展示拟合曲线与测试误差
- 下排三幅柱状图展示 21 个系数的幅值分布——直观对比 Lasso 的稀疏性与 Ridge 的收缩性

---

### 04 — 梯度下降法

#### 4.1 单变量二次函数优化 (`04_gradient_desent1.ipynb`)

目标函数：$f(x) = (x + 3)^2 - 5$

- 学习率 $\alpha = 0.1$，初始点 $x_0 = 1$
- 解析计算梯度 $\nabla f(x) = 2(x + 3)$
- 迭代 100 次，从 $x=1$ 收敛至 $x=-3$（全局最小值，$f(-3) = -5$）
- 可视化：在函数曲线上叠加红色轨迹点，展示参数更新的完整路径

#### 4.2 非凸函数优化 (`04_gradient_desent2.ipynb`)

目标函数：$J(x) = (x^2 - 2)^2$

- 具有两个对称的全局最小值 $x = \pm\sqrt{2} \approx \pm 1.414$
- 梯度 $\nabla J(x) = 4x(x^2 - 2)$
- 学习率 $\alpha = 0.01$，使用 `while` 循环以梯度范数 $< 10^{-10}$ 为收敛判据
- 全局视图 + 局部放大视图双面板对比，展示在最小值附近的精细收敛行为

---

### 05 — 分类模型评估指标

#### 5.1 混淆矩阵 (`05_classification_test.ipynb`)

以猫/狗二分类为例，手动构建真实标签与预测标签，系统推导：

| 指标 | 公式 | 含义 |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 |
| **Precision (查准率)** | $\frac{TP}{TP + FP}$ | 预测为正的样本中真正为正的比例 |
| **Recall (查全率)** | $\frac{TP}{TP + FN}$ | 真实正样本中被正确识别的比例 |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Precision 和 Recall 的调和平均 |

使用 `seaborn.heatmap` 绘制混淆矩阵，配合 `classification_report` 输出完整的分类报告。

#### 5.2 分类报告与 ROC 曲线 (`05_classification_report.py`)

- 使用 `make_classification` 生成 1000 样本 × 20 特征的二分类合成数据
- 训练逻辑回归模型，输出 4 位精度的分类报告
- 计算 AUC (Area Under Curve) 值
- 绘制 ROC 曲线：TPR vs FPR，以随机猜测对角线为基线参考

---

### 06 — K-近邻 (KNN)

#### 6.1 KNN 分类器 (`06_knn_classification.ipynb`)

- 4 个训练样本，2 个类别，2 维特征空间
- `KNeighborsClassifier(n_neighbors=2, weights="distance")` — 距离加权投票
- 预测新样本 `(4, 9)` 的类别
- 使用布尔索引进行类别着色，散点图展示决策结果

#### 6.2 KNN 回归器 (`06_knn_regression.ipynb`)

- 相同数据，切换为回归任务：`KNeighborsRegressor(n_neighbors=2)`
- 预测值为最近邻目标值的均值：`(1 + 0) / 2 = 0.5`

#### 6.3 特征缩放 (`06_knn_scaler.ipynb`)

KNN 对特征尺度敏感——量纲差异会导致距离计算失真。对比两种缩放策略：

| 方法 | 缩放范围 | 适用场景 |
|---|---|---|
| **MinMaxScaler** `(-1, 1)` | 固定区间映射 | 有界输入，保留稀疏性 |
| **StandardScaler** | 均值 0，标准差 1 | 假定高斯分布，对异常值更鲁棒 |

---

### 07 — 心脏病预测：KNN + 网格搜索

**文件:** `07_heart_disease.py`

完整的有监督学习 Pipeline：

```
数据加载 → 缺失值处理 → 特征分类 → ColumnTransformer →
标准化 (数值) + 独热编码 (类别) + 直通 (二元) → KNN + GridSearchCV
```

**特征工程策略：**

| 特征类型 | 特征名 | 处理方法 |
|---|---|---|
| 数值型 | `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca` | `StandardScaler` 标准化 |
| 类别型 | `cp` (胸痛类型), `restecg`, `slope`, `thal` | `OneHotEncoder` 独热编码 |
| 二元型 | `sex`, `fbs`, `exang` | `passthrough` 直接保留 |

**超参数调优：**

```python
param_grid = {
    "n_neighbors": range(1, 11),    # k 从 1 到 10
    "weights": ["uniform", "distance"]  # 均匀权重 vs 距离加权
}
```

使用 `GridSearchCV(cv=10, scoring="accuracy")` 进行 10 折交叉验证搜索最佳超参数组合。

---

### 08 — 线性回归

#### 8.1 Scikit-learn 实现 (`08_linear_regression.ipynb`)

- 问题：用每周学习时长预测考试成绩
- `LinearRegression()` 最小二乘拟合：$\hat{y} = 2.87x + 41.45$
- 对比 `fit_intercept=True`（有截距）与 `fit_intercept=False`（强制过原点）的效果差异
- 可视化：散点图叠加回归直线

#### 8.2 梯度下降手写实现 (`08_liner_regression_gradient.py`)

不依赖 Scikit-learn，从零实现线性回归的梯度下降求解：

- 损失函数（MSE）：$J(\beta) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$
- 梯度向量：$\nabla J(\beta) = \frac{2}{n}X^T(X\beta - y)$
- 设计矩阵：添加全 1 列作为截距项（`np.hstack`）
- 参数更新：$\beta := \beta - \alpha \nabla J(\beta)$
- 收敛条件：梯度各分量绝对值均 $< 10^{-10}$ 且迭代次数未耗尽
- 可视化：$\beta_0$-$\beta_1$ 参数空间中的更新轨迹

#### 8.3 SGDRegressor (`08_sgd.ipynb`)

使用随机梯度下降求解相同问题：

```python
SGDRegressor(
    penalty=None,               # 无正则化
    loss='squared_error',       # 均方误差损失
    eta0=1e-7,                  # 极小的恒定学习率
    learning_rate='constant',
    tol=1e-8                    # 收敛容忍度
)
```

> 注意：SGD 收敛对学习率高度敏感——过大发散，过小收敛极慢。

---

### 09 — 广告销量预测：多元线性回归

**文件:** `09_advertising/advertising.py`

使用 Advertising 数据集（TV / Radio / Newspaper 广告支出 → Sales 销量），对比两种线性模型的性能：

| 模型 | 特点 |
|---|---|
| `LinearRegression` | 正规方程解析解，精确且快速（小数据集） |
| `SGDRegressor` | 随机梯度下降迭代解，适合大规模/在线学习 |

**Pipeline:**
```
读取 CSV → 去除缺失值 → 分离特征与目标 →
StandardScaler 标准化 → 训练两个模型 → MSE 评估对比
```

`StandardScaler` 对 SGD 尤为重要——未标准化的特征会破坏梯度下降的收敛性。

---

### 10 — 逻辑回归

#### 10.1 API 参数探索 (`10_logistic_regression/1_api_test.ipynb`)

测试逻辑回归的关键超参数：

| 参数 | 含义 | 取值示例 |
|---|---|---|
| `solver` | 优化算法 | `'sag'` (随机平均梯度) |
| `penalty` | 正则化类型 | `'l1'` (Lasso), `'l2'` (Ridge) |
| `C` | 正则化强度倒数 | `1` (越小正则化越强) |
| `class_weight` | 类别权重 | `'balanced'` (自动平衡) |
| `max_iter` | 最大迭代次数 | `1000` |

#### 10.2 手写数字识别 (`10_logistic_regression/3_digit_recognizer.py`)

MNIST 数据集上的多分类逻辑回归：

- 输入：784 维像素特征（28×28 灰度图）
- 预处理：`MinMaxScaler` 归一化至 [0, 1]
- 模型：`LogisticRegression(max_iter=500)`（默认 one-vs-rest 多分类）
- 预测：输出类别标签及概率分布（`predict_proba`）
- 可视化：`imshow(cmap='gray')` 展示手写数字图像

#### 10.3 心脏病预测 (`10_logistic_regression/heart_disease.py`)

使用与 07 相同的特征工程 Pipeline（`ColumnTransformer`），将分类器从 KNN 替换为逻辑回归，验证线性分类边界在结构化医疗数据上的表现。

---

### 12 — 感知机与逻辑门

**文件:** `12_perceptron/1_logic_gate.py`

从零实现感知机模型（单层人工神经网络），展示其计算能力与局限性：

| 逻辑门 | 权重 | 偏置 | 线性可分？ |
|---|---|---|---|
| **AND** | $[0.5, 0.5]$ | $-0.7$ | ✅ 是 |
| **OR** | $[0.5, 0.5]$ | $-0.2$ | ✅ 是 |
| **NAND** | $[-0.5, -0.5]$ | $0.7$ | ✅ 是 |
| **XOR** | 组合 NAND + OR + AND | — | ❌ 否（需多层） |

**核心思想：**

- 感知机决策：$y = \begin{cases} 1 & \text{if } \mathbf{w}^T\mathbf{x} + b > 0 \\ 0 & \text{otherwise} \end{cases}$
- XOR 不能由单层感知机表示（非线性可分），通过叠加 NAND → OR → AND 两层结构实现——这正是多层神经网络的基本原理

---

### 13 — 信息熵

**文件:** `13_supervise/1_entropy.ipynb`

可视化二元随机变量的香农信息熵：

$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

- 当 $p = 0.5$（完全不确定）时，熵取最大值 1 bit
- 当 $p \to 0$ 或 $p \to 1$（完全确定）时，熵趋近于 0
- 这是决策树分裂准则（信息增益）和交叉熵损失函数的理论基础

---

## 学习路线建议

```
01 NumPy 基础
    │
    ▼
02 多项式拟合 ──→ 03 正则化
    │
    ▼
04 梯度下降 ──→ 08 线性回归 ──→ 09 多元线性回归
    │                              │
    ▼                              ▼
05 分类评估 ──→ 10 逻辑回归 ──→ 手写数字识别
    │
    ▼
06 KNN ──→ 07 KNN + 网格搜索
    │
    ▼
12 感知机 ──→ 13 信息熵
```

---

## 参考资料

- [Scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [NumPy 官方文档](https://numpy.org/doc/stable/)
- [Pattern Recognition and Machine Learning — Christopher Bishop](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [The Elements of Statistical Learning — Hastie, Tibshirani, Friedman](https://hastie.su.domains/ElemStatLearn/)
