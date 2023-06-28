# Classical machine learning. Learning with a teacher. Regression. 
# Linear regression. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression

"""
## Лінійна регресія з однією змінною (h)
h(x) = w_0 + w_1 * x  #  залежність від одного параметру

## Лінійна регресія з багатьма змінними (h = linear_regression_hypothesis)
h(x) = w⋅x  # залежність від багатьох параметрів
h(x) = w_0 * (x0:=1) + w_1 * x1 + w_2 * x2 + ... w_m * xm
"""

full_df = pd.read_csv('helps/Housing.csv')
new_df = full_df.iloc[:, 0:4]
# Розв'язання задачі лінійної регресії за допомогою sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
X = np.array([new_df.area, new_df.bedrooms, new_df.bathrooms]).T
y = np.array(new_df.price)
reg = LinearRegression().fit(X, y)

# intercept_  =>  Independent term in the linear model  (w_0)
# coef_  =>  Estimated coefficients for the linear regression problem. (w_1, w_2, ... w_i)
# score(X, y[, sample_weight])  =>  Return the coefficient of determination of the prediction.
print(reg.intercept_, reg.coef_, reg.score(X, y))




# Logistic regression.

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def plot_data(X, y, pos_label='y=1', neg_label='y=0'):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)


X_train, y_train = load_data('helps/data.txt')

plt.figure(figsize=(8, 6))
plot_data(X_train, y_train)
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc='upper right')
plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)
# model.coef_
# model.intercept_

"""
w0 + w1 * x1 + w2 * x2 = 0
y = k * x + b
x2 = -w1 * x1 / w2 - w0 / w2
"""

def decision_boundary(model: LogisticRegression, x: np.array) -> np.array:
    w_0 = model.intercept_[0]
    w_1 = model.coef_[0][0]
    w_2 = model.coef_[0][1]

    return -w_1 * x / w_2 - w_0 / w_2


x = np.linspace(5, 100, 100)

plt.figure(figsize=(8, 6))
plot_data(X_train, y_train)
plt.plot(x, decision_boundary(model, x), label='decision boundary')
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc='upper right')
plt.show()


x = np.linspace(5, 100, 100)
y = np.linspace(5, 100, 100).T
xx, yy = np.meshgrid(x, y)

X= np.c_[xx.ravel(), yy.ravel()]

probas = model.predict_proba(X)
pr = probas[:, 1].reshape((100, 100))


plt.figure(figsize= (10, 10))

imshow_handle = plt.imshow(pr, origin='lower')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title('Probability')
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
plt.show()

"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=0).fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])

clf.score(X, y)
"""
