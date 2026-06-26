import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


# 1.读取数据
dataset = pd.read_csv(r"D:\桌面\machine_learning\archive\heart.csv")
dataset.dropna(inplace=True)

# 2.划分数据集
X = dataset.drop('target', axis=1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程
# 3.1 数值特征标准化
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak","ca"]
# 3.2 类别特征独热编码
categorical_features = ["cp", "restecg", "slope", "thal"]
# 3.3 二元特征
binary_features = ["sex", "fbs", "exang"]

# 创建一个ColumnTransformer对象
columnTransformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)
# 3.4 对训练集和测试集进行特征转换
x_train = columnTransformer.fit_transform(X_train)
x_test = columnTransformer.transform(X_test)

print(x_train.shape, x_test.shape)

# 4. 模型训练
model = LogisticRegression(
    
)
model.fit(x_train, y_train)

# 5. 模型评估
print(model.score(x_test, y_test))