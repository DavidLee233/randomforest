import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pydotplus
from IPython.display import Image

# 显示前几个数字图像
def show_image(data):
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8,8,i+1, xticks=[], yticks=[])
        ax.imshow(data.images[i], cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0,7,str(data.target[i]))
    fig.show()


# GridSearchCV网络搜索交叉验证，用来遍历找出最合适的参数
def GridSearch(train_data,train_target):
    tree_param_grid = { 'max_features':list((20, 30, 40, 50, 60)),
                        'min_samples_split':list(range(3, 8)),
                        'n_estimators':list((100, 200, 300, 400, 500)),
                        'max_depth':list((5, 6, 7, 8, 9, 10))}
    grid = GridSearchCV(RandomForestClassifier(criterion='gini', random_state=100),param_grid=tree_param_grid, cv=3)
# 先指定算法，然后把想找的参数写成字典传给param_grid，cv是交叉验证次数
    grid.fit(train_data, train_target)
    print(grid.cv_results_)   # 查看训练详细结果
    print(grid.best_params_)  # 看最终得到的参数
    print(grid.best_score_)   # 查看遍历过程中最高的score


# 画出混淆矩阵
def show_confusion_matrix(test_target, prediction_target):
    mat = confusion_matrix(test_target, prediction_target)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


# 可视化随机森林，即循环可视化决策树
def draw_tree(model, data):
    for idx, estimator in enumerate(model.estimators_):
        dot_data= tree.export_graphviz(
            estimator,#改成自己的树
            out_file = None,
            feature_names = np.array(data.feature_names),#改成需要的特征名
            filled = True,
            impurity = True,#是否显示gini系数或熵值
            rounded = True,
            special_characters=True,
            class_names=str(data.target)
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[1].set_fillcolor("#2DF3DF")
    Image(graph.create_png())
    graph.write_pdf("tree_num{}.pdf".format(idx))#保存图片



def test_criterion(train_data, train_target, test_data, test_target):
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
    max_tree_number = 200  # 最大树的数量(超参数上限)
    for i in range(10, max_tree_number, 10):
        clf1 = RandomForestClassifier(criterion="gini", max_depth=9, random_state=100, n_estimators=i)
        clf2 = RandomForestClassifier(criterion="entropy", max_depth=9, random_state=100, n_estimators=i)
        clf1 = clf1.fit(train_data, train_target)  # 训练模型
        clf2 = clf2.fit(train_data, train_target)
        score_test1 = clf1.score(test_data, test_target)  # 测试集预测准确率
        score_test2 = clf2.score(test_data, test_target)
        test_score_list1.append(score_test1)
        test_score_list2.append(score_test2)
        clf_list1.append(clf1)
        clf_list2.append(clf2)
        print('当前树的数量：', i, '当前数量下的gini纯度指标下测试集验证准确率：', score_test1, '当前数量下的entropy纯度指标下测试集验证准确率：', score_test2)
    print('最高gini准确率为:', max(test_score_list1), '最高gini准确率对应树的数量为：', 10 * (test_score_list1.index(max(test_score_list1)) + 1))
    print('最高entropy准确率为:', max(test_score_list2), '最高entropy准确率对应树的数量为：', 10 * (test_score_list2.index(max(test_score_list2)) + 1))
    plt.plot(range(1, 20), test_score_list1, color="red", label="gini-test")
    plt.plot(range(1, 20), test_score_list2, color="blue", label="entropy-test")
    plt.xlabel('决策树深度', fontproperties="SimSun")
    plt.ylabel('准确率', fontproperties="SimSun")
    plt.title('两种纯度指标下的测试集预测准确率比较', fontproperties="SimSun")
    plt.legend()
    plt.show()
    i = test_score_list1.index(max(test_score_list1))
    j = test_score_list2.index(max(test_score_list2))
    # 最优模型
    rfc1 = clf_list1[i]     # 选取测试集预测准确率最大值的模型#
    rfc2 = clf_list2[j]
    #预测测试集并评估模型
    y_pre1 = rfc1.predict(test_data)
    y_pre2 = rfc2.predict(test_data)
    print(metrics.classification_report(y_pre1, test_target))
    print(metrics.classification_report(y_pre2, test_target))
    test_right1 = np.count_nonzero(y_pre1 == test_target)  # 统计预测正确的个数
    test_right2 = np.count_nonzero(y_pre2 == test_target)
    print('gini纯度指标下的预测正确数目：', test_right1)
    print('准确率: %.2f%%' % (100 * float(test_right1) / float(len(test_target))))
    print('entropy纯度指标下预测正确数目：', test_right2)
    print('准确率: %.2f%%' % (100 * float(test_right2) / float(len(test_target))))

def cross_validation(data):
    score = []
    for i in range(0, 200, 10):
        model = RandomForestClassifier(n_estimators=i + 1, random_state=100, criterion='gini', max_depth=9)
        score.append(cross_val_score(model, data.data, data.target, cv=10).mean())
        print('当前树的数量：', i, '当前数量下交叉验证准确率：',cross_val_score(model, data.data, data.target, cv=10).mean())
    print('最高准确率为:', max(score), '最高准确率对应树的数量为：', 10*(score.index(max(score))+1))
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 21), score)
    plt.title('决策树数量\准确率曲线', fontproperties="SimSun")
    plt.xlabel('树的数量', fontproperties="SimSun")
    plt.ylabel('准确率', fontproperties="SimSun")
    plt.show()