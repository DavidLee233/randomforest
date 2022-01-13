import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
digits = load_digits()
# 设置图形对象
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# 画数字：每个数字是8X8像素
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # 用target值给图像做标注
    ax.text(0, 7, str(digits.target[i]))
plt.show()

# 将数据集分为训练集和测试集
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0, test_size=0.3)
# 建立模型，参数设置成1000，代表由1000个随机决策树组合而成
model = RandomForestClassifier(n_estimators=2, criterion='gini')
# 拟合模型
model.fit(Xtrain, ytrain)
# 预测测试集
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))