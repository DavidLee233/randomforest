from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import  numpy as np

def best_forest(train_data, train_target, test_data, test_target):
    """
    调参得到最佳的max_depth值并返回对应训练后的模型
    :param train: 训练集
    :param test: 测试集
    :return: 训练后的模型列表和测试集预测准确率最大值的索引
    """
    test_score_list1 = []
    test_score_list2 = []
    clf_list1 = []
    clf_list2 = []
    max__tree_number = 50  # 最大树深(超参数上限)
    for i in range(0, max__tree_number):
        clf1 = RandomForestClassifier(criterion="gini", max_depth=5, random_state=30, n_estimators=i+1)
        clf2 = RandomForestClassifier(criterion="entropy", max_depth=5, random_state=30, n_estimators=i+1)
        clf1 = clf1.fit(train_data, train_target)  # 训练模型
        clf2 = clf2.fit(train_data, train_target)
        score_test1 = clf1.score(test_data, test_target)  # 测试集预测准确率
        score_test2 = clf2.score(test_data, test_target)
        test_score_list1.append(score_test1)
        test_score_list2.append(score_test2)
        clf_list1.append(clf1)
        clf_list2.append(clf2)
    plt.plot(range(1, max__tree_number + 1), test_score_list1, color="red", label="gini-test")
    plt.plot(range(1, max__tree_number + 1), test_score_list2, color="blue", label="entropy-test")
    plt.xlabel('决策树深度', fontproperties="SimSun")
    plt.ylabel('准确率', fontproperties="SimSun")
    plt.title('两种纯度指标下的测试集预测准确率比较', fontproperties="SimSun")
    plt.legend()
    plt.show()
    return clf_list1, clf_list2, test_score_list1.index(max(test_score_list1)), test_score_list2.index(
        max(test_score_list2))

# 加载sklearn包手写数据集
digit = load_digits()
features = digit.data
target = digit.target
# 划分训练集和测试集
# 定义训练集占70%，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

# 训练决策树分类器对象
clf_list1, clf_list2, i, j =  best_forest(X_train, y_train, X_test, y_test)      # 训练模型
# 最优模型
model1 = clf_list1[i]     # 选取测试集预测准确率最大值的模型#
model2 = clf_list2[j]
#预测测试集并评估模型
y_pre1 = model1.predict(X_test)
y_pre2 = model1.predict(X_test)
print(metrics.classification_report(y_pre1, y_test))
print(metrics.classification_report(y_pre2, y_test))
test_right1 = np.count_nonzero(y_pre1 == y_test)  # 统计预测正确的个数
test_right2 = np.count_nonzero(y_pre2 == y_test)
print('gini纯度指标下的预测正确数目：', test_right1)
print('准确率: %.2f%%' % (100 * float(test_right1) / float(len(y_test))))
print('entropy纯度指标下预测正确数目：', test_right2)
print('准确率: %.2f%%' % (100 * float(test_right2) / float(len(y_test))))

