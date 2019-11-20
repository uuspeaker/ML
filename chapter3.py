from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

mnist = load_mnist()
(X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=False)
print(X_train.shape, X_test.shape)

import matplotlib
import matplotlib.pyplot as plt
some_digit = X_train[123]
some_digit_image = some_digit.reshape(28, 28)
print(y_train[123])
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
# plt.axis("off")
# plt.show()

import numpy as np
shuffle_index = np.random.permutation(30000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)
result = sgd_clf.predict([some_digit])
print(result)

# 交叉验证法计算准确率
from sklearn.model_selection import cross_val_score
cross_val_score1 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("cross_val_score_normal",cross_val_score1)

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score2 = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("cross_val_score_never5",cross_val_score2)

# 计算混淆矩阵
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(y_train_5, y_train_pred)
print("confusion_matrix",confusion_matrix1)

# 计算查准率,查全率,F1
from sklearn.metrics import precision_score, recall_score
precision_score1 = precision_score(y_train_5, y_train_pred)
print("precision_score1",precision_score1)
recall_score1 = recall_score(y_train_5, y_train_pred)
print("recall_score1",recall_score1)

from sklearn.metrics import f1_score
f1_score = f1_score(y_train_5, y_train_pred)
print("f1_score",f1_score)

y_scores = sgd_clf.decision_function([some_digit])
print("y_scores",y_scores)
threshold = -200000
y_some_digit_pred = (y_scores > threshold)
print("y_some_digit_pred--200000",y_some_digit_pred)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print("y_some_digit_pred-0",y_some_digit_pred)

# 计算决策分数
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
method="decision_function")

from sklearn.metrics import precision_recall_curve
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# 调整阀值以便调整查准率和查全率
y_train_pred_90 = (y_scores > -500000)
precision_score90 = precision_score(y_train_5, y_train_pred_90)
print("precision_score90",precision_score90)
recall_score90 = recall_score(y_train_5, y_train_pred_90)
print("recall_score90",recall_score90)

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
some_digit_scores = sgd_clf.decision_function([some_digit])
print("some_digit_scores",some_digit_scores)

max = np.argmax(some_digit_scores)
print("max",max)
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

# 随即森林的多分类
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_pro = forest_clf.predict_proba([some_digit])
print("forest_pro",forest_pro)
forest_accuracy = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("forest_accuracy",forest_accuracy)

# 归一化提升准确率
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print("cross_val_score_scaled",cross_val_score_scaled)

# 分析错误
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print("conf_mx",conf_mx)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()


# 察看原始数据
import matplotlib as mpl
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

# 多标签分类
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf_pre = knn_clf.predict([some_digit])
print("knn_clf_pre",knn_clf_pre)

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
knn_f1_score = f1_score(y_train, y_train_knn_pred, average="macro")
print("knn_f1_score",knn_f1_score)

# 多输出分类
noise = rnd.randint(0, 100, (len(X_train), 784))
noise = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit_moise = plot_digit(clean_digit)
print("plot_digit_moise",plot_digit_moise)