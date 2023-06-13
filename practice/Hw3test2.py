# %matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


def linear_regression_hypothesis(w: np.array, x: np.array) -> float:
    """
    Return the value of the linear regression hypothesis of dependence on many parameters.
    w: weights;
    x: parameters, where x[0] = 1.
    """
    if len(w) != len(x) or x[0] != 1:
        return None
    
    return np.dot(w, x)


df = pd.read_csv('practice/Housing.csv')
df.head(3)

# prepare df:
new_df = df.iloc[:, 0:4]
new_df.head(3)


x = pd.DataFrame()
x[0] = np.ones((new_df.shape[0],), dtype=int)
for num in range(1, new_df.shape[1]):
    x[num] = df.iloc[:, num]

x.head(3)


def loss_function_many(w: np.array, df: pd.DataFrame, x: pd.DataFrame) -> float:
    """Calculation of the loss function in vector form (for dependence on many parameters).
    w: weights;
    df: DataFrame, where 0-column is y, last each one - important parameters;
    x: DataFrame of parameters where 0 column is 1 (x[0]=1).
    """
    if len(w) != x.shape[1]:
        return None
    
    n: int = df.shape[1]
    m: int = df.shape[0]
        
    cost = 0
    for line_idx_y, y in enumerate(df.iloc[:, 0]):
        cost += (y - linear_regression_hypothesis(w, x.iloc[line_idx_y])) ** 2

    return cost/(2*m)


def grad_step_many(weights: list, grads: list, learning_rate: float = 0.001) -> list:
    """Function of one step of gradient descent with many parameters. Return the weights (list)."""
    # weights = list(weights)
    # grads = list(grads)
    weights = [weights[num] - learning_rate * grads[num] for num in range(len(weights))]
    
    return weights


def grad_w_i(weights: list, df: pd.DataFrame, x: pd.DataFrame, i: int) -> float:
    """Calculation of the graduation descent weight 1 (for dependence on one parameter).
    weights is list of values from vector weights;
    df: DataFrame;
    x: DataFrame of parameters where 0 column is 1 (x[0]=1);
    i: number of i-parameter in x.
    """
    n: int = df.shape[0]
    cost = 0

    for line_idx_y, y in enumerate(df.iloc[:, 0]):
        cost += (y - linear_regression_hypothesis(weights, x.iloc[line_idx_y])) * x.iloc[line_idx_y, i]

    return cost/n



def grad_descent(
                 weights: list, 
                 df: pd.DataFrame, 
                 num_iter: int, 
                 x: pd.DataFrame,
                 learning_rate: float = 0.001, 
                 epsilon: float = 0.0000001,
                 ) -> tuple:
    """Gradient descent function with one parameter. Return weights and story of the descent."""
    loss_history = [loss_function_many(weights, df, x)]
    for _ in range(num_iter):
        grads = [grad_w_i(weights, df, x, el) for el in range(x.shape[1])]
        weights = grad_step_many(weights, grads, learning_rate=learning_rate)

        loss = loss_function_many(weights, df, x)
        loss_history.append(loss)

        if abs(loss - loss_history[-2]) < epsilon:
            break

    return (weights, loss_history)


# weights = [0. for _ in range(x.shape[1])]
# weights, history = grad_descent(weights, df=new_df, num_iter=10000, learning_rate=0.01, x=x)
# print(weights, '\n', history)


def normalization(data) -> list:
    """Return normalized values (list) of the array_like object (data)."""
    mean: float = np.mean(data)
    value_range = np.max(data) - np.min(data)

    return [(x - mean) / value_range for x in data]


new_df = pd.DataFrame()
new_df['price'] = normalization(df.price)
new_df['area'] = normalization(df.area)
new_df['bedrooms'] = normalization(df.bedrooms)
new_df['bathrooms'] = normalization(df.bathrooms)
print(new_df.head(3))
x = pd.DataFrame()
x[0] = np.ones((new_df.shape[0],), dtype=int)
for num in range(1, new_df.shape[1]):
    x[num] = new_df.iloc[:, num]

x.head(3)


def x_prepare(w, x):
    return [x if i != 0 else 1.0 for i in range(len(w))]
    # return np.asarray([x if i != 0 else 1.0 for i in range(len(w))], dtype='float')



weights = [0. for _ in range(x.shape[1])]
weights, history = grad_descent(weights, df=new_df, num_iter=1000, learning_rate=0.01, x=x)
print(weights, '\n', history)



X = np.array([new_df.area, new_df.bedrooms, new_df.bathrooms]).T
y = np.array(new_df.price)
reg = LinearRegression().fit(X, y)


plt.figure(figsize=(6, 6))
plt.plot(new_df.area, new_df.price, 'go')
x = np.linspace(np.min(new_df), np.max(new_df), 3)
y = [linear_regression_hypothesis(weights, x_prepare(weights, xi)) for xi in x]
plt.plot(
         x, 
         y, 
         label='An.'
         )
y2 = [linear_regression_hypothesis((reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2]), (1, xi, xi, xi)) for xi in x]
plt.plot(
         x,
         y2,
         label='LinearRegression',
         )
plt.legend()
plt.show()


