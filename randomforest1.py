from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


def show_img(imgs):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(imgs.images[i], cmap='gray', interpolation="nearest")
        ax.text(0, 7, str(imgs.target[i]))
    fig.show()


def show_matrix(pre, target):
    mat = metrics.confusion_matrix(pre, target)
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.show()

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=100)
model = RandomForestClassifier(n_estimators=50, random_state=90, criterion='entropy')
model.fit(x_train, y_train)
pre = model.predict(x_test)
print(metrics.classification_report(pre, y_test))
print(metrics.accuracy_score(pre, y_test))

score = []
for i in range(0, 100, 10):
    model = RandomForestClassifier(n_estimators=i+1, random_state=90)
    score.append(cross_val_score(model, digits.data, digits.target, cv=10).mean())
    print(i, cross_val_score(model, digits.data, digits.target, cv=10).mean())
print(max(score), score.index(max(score)))
plt.figure(figsize=[20, 5])
plt.plot(range(1, 11), score)
plt.show()