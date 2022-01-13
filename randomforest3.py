from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utile import *
# 加载sklearn包手写数据集
digit = load_digits()
features = digit.data
target = digit.target
# 划分训练集和测试集
# 定义训练集占70%，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=5)

# 可视化数据集
show_image(digit)

# GridSearchCV网格搜索交叉验证，用来遍历找出最合适的参数
GridSearch(X_train,y_train)


# 交叉验证法得到最佳模型
# cross_validation(digit)

# 训练得到两种不同纯度指标下的准确率比较
# test_criterion(X_train, y_train, X_test, y_test)

# 开始训练最优参数下的随机森林
rfc=RandomForestClassifier(criterion="gini", max_depth=9, random_state=100, n_estimators=300)
rfc=rfc.fit(X_train,y_train)
result=rfc.score(X_test,y_test)
print('预测准确率为：', result)


# 查看特征重要程度
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]  # a[::-1]让a逆序输出
print('按维度重要性排序的维度的序号：', indices)

#查看分类器的分类结果报告
y_pre = rfc.predict(X_test)
print(metrics.classification_report(y_pre, y_test))

# 画出混淆矩阵
show_confusion_matrix(y_test,y_pre)

# 可视化随机森林
# draw_tree(rfc, digit) # 绘制决策树


