"""
1.age
2.sex
3.chest pain type (4 values)
4.resting blood pressure
5.serum cholestoral in mg/dl
6.fasting blood sugar > 120 mg/dl
7.resting electrocardiographic results (values 0,1,2)
8.maximum heart rate achieved
9.exercise induced angina
10.oldpeak = ST depression induced by exercise relative to rest
11.the slope of the peak exercise ST segment
12.number of major vessels (0-3) colored by flourosopy
13.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect    
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# 1.读取heart.csv文件
heart_disease_data = pd.read_csv('./archive/heart.csv')

# 查看前五行数据
# print(heart_disease_data.head())

# 处理缺失值
heart_disease_data.dropna(inplace=True)

heart_disease_data.info()
print(heart_disease_data.head())

# 2. 划分数据集
# 划分特征和标签
X = heart_disease_data.drop("target", axis=1)
y = heart_disease_data["target"]

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

# 特征转换
X_train = columnTransformer.fit_transform(X_train)
X_test = columnTransformer.transform(X_test)
print(X_train.shape, X_test.shape)

"""
# 4.创建模型
knn = KNeighborsClassifier(n_neighbors=3)

# 5. 模型训练
knn.fit(X_train, y_train)

# 6. 模型评估
accuracy = knn.score(X_test, y_test)
print(f"模型准确率: {accuracy}")

# 7. 模型保存
joblib.dump(knn, "heart_disease_knn_model")
"""

# 创建knn分类器
knn = KNeighborsClassifier()

# 定义网格搜索参数列表
param_grid = {
    "n_neighbors": list(range(1, 11)),
    "weights": ["uniform", "distance"]
}

grid_search = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
grid_search.fit(X_train, y_train)

# 8. 输出最佳参数和最佳准确率
results = pd.DataFrame(grid_search.cv_results_)
# print(results)

# 9. 输出最佳参数和最佳准确率
print("最佳模型:", grid_search.best_estimator_)
print("最佳参数:", grid_search.best_params_)
print("最佳准确率:", grid_search.best_score_)

# 使用最佳模型进行预测
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# 评估模型
accuracy = best_knn.score(X_test, y_test)
print(f"最佳模型准确率: {accuracy}")