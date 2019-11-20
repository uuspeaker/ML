# 闭式解回归
import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("theta_best",theta_best)

X_new = np.array([[0], [2]])
print("X_new",X_new)
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
print("X_new_b",X_new_b)
y_predict = X_new_b.dot(theta_best)
print("y_predict",y_predict)
# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

# 线性回归
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
X_new_pre = lin_reg.predict(X_new)
print("X_new_pre",X_new_pre)

# 批量梯度下降法
eta = 0.1 # learning rate
n_epochs = 50
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
print("theta",theta)

# 随机梯度下降
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1) # random initialization
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print("theta_random",theta)

# 调用系统方法随机梯度下降
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
print("sgd_reg.intercept_, sgd_reg.coef_",sgd_reg.intercept_, sgd_reg.coef_)

# 多项式回归
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print("X_poly[0]",X_poly[0])
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print("lin_reg.intercept_, lin_reg.coef_",lin_reg.intercept_, lin_reg.coef_)

# 学习曲线
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    plt.show()

lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)

# 10阶多项式学习曲线
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
("poly_features", PolynomialFeatures(degree=4, include_bias=False)),
("sgd_reg", LinearRegression()),
))
# plot_learning_curves(polynomial_regression, X, y)

# 闭式解岭回归
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg_result = ridge_reg.predict([[1.5]])
print("ridge_reg.intercept_, ridge_reg.coef_",ridge_reg.intercept_, ridge_reg.coef_)
print("ridge_reg_result",ridge_reg_result)

sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l2", eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg_r = sgd_reg.predict([[1.5]])
print("sgd_reg_r",sgd_reg_r)
# plt.plot(X, y, "b.")
# plt.show()

# 套索回归
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg_r = lasso_reg.predict([[1.5]])
print("lasso_line",lasso_reg_r)

sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l1", eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg_r = sgd_reg.predict([[1.5]])
print("sgd_lasso",sgd_reg_r)

# 弹性网络
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net_result = elastic_net.predict([[1.5]])
print("elastic_net_result",elastic_net_result)

# 早期停止法
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

#
from sklearn import datasets
iris = datasets.load_iris()
print("iris.keys",list(iris.keys()))
X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0
# print("iris X, y", X, y)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
# plt.show()
result = log_reg.predict([[1.7], [1.5]])
print("result",result)

# 逻辑回归
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
r1 = softmax_reg.predict([[5, 2]])
print("r1",r1)
r2 = softmax_reg.predict_proba([[5, 2]])
print("r2",r2)