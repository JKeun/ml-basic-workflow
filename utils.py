import math
import random
import numpy as np
import pandas as pd
import pydotplus
import random

from six import StringIO
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image


# math
def sqr(x):
    return x*x

def sqrt(x):
    if x >= 0:
        return math.sqrt(x)
    else:
        return None
    
def mean(xs):
    res = 0
    for x in xs:
        res += x
    return res/len(xs)


# data generation
def generate_gaussian_vector(size, mu=0, sigma=1):
    res = [0] * size
    for k in range(size):
        res[k] = random.gauss(mu, sigma)
    return res

def make_linear_dataset(num_samples=1000,
                        num_features=20,
                        target_noise_sigma=5,
                        add_plus_noise=False,
                        seed=42
                        ):
    np.random.seed(seed)
    
    X = np.random.uniform(10, 50, size=(num_samples, num_features))
    w = np.random.randn(num_features)
    y = np.dot(X, w) + target_noise_sigma * np.random.randn(num_samples)
    
    if add_plus_noise:
        for _ in range(5):
            y[np.random.randint(num_samples)] += 10000 * np.random.rand()
    
    return X, y

def make_non_linear_dataset(num_samples=1000,
                            num_features=20,
                            target_noise_sigma=5,
                            seed=42
                            ):
    np.random.seed(42)
    
    X = np.random.uniform(10, 50, size=(num_samples, num_features))
    w = np.random.randn(num_features)
    
    # square some random input features
    sqr_idxs = np.random.choice(range(num_features), int(num_features / 2))
    X[:, sqr_idxs] = X[:, sqr_idxs] ** 2
    
    # sine some random input features
    sin_idxs = np.random.choice(range(num_features), int(num_features / 2))
    X[:, sin_idxs] = np.sin(X[:, sin_idxs])
    
    y = np.sqrt(np.abs(np.dot(X, w))) + target_noise_sigma * np.random.randn(num_samples)
    
    return X, y


# linear models
def solve(X, y, alpha=0):
    X_t = X.T
    X_t_X = np.dot(X_t, X)
    X_t_y = np.dot(X_t, y)
    X_t_X_inv = np.linalg.inv(X_t_X + alpha * np.eye(len(X_t)))
    w_hat = np.dot(X_t_X_inv, X_t_y)
    return w_hat

def solve_predict_norm(X, y, alpha=0):
    mean_Xs, sd_Xs = np.mean(X, axis=0), np.std(X, axis=0)
    mean_y, sd_y = np.mean(y), np.std(y)
    
    # normalize
    norm_Xs = (X - mean_Xs) / sd_Xs
    norm_ys = (y - mean_y) / sd_y
    
    # solve
    norm_w_hat = solve(norm_Xs, norm_ys, alpha)
    norm_y_hat = np.dot(norm_Xs, norm_w_hat)
    
    # denormalize, return to real scale
    y_hat = mean_y + sd_y * norm_y_hat
    return y_hat, norm_w_hat

def ridge(X, y, alpha=0.01):
    return solve(X, y, alpha=alpha)

def norm_ridge(X, y, alpha=0.01):
    return solve_predict_norm(X, y, alpha=alpha)


# tree model
def make_trained_decision_tree(X, y, depth=4):
    
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X, y)
    return tree

def decision_tree_predict(tree, X):
    return tree.predict(X)

def get_tree_image(dt):
    dot_data = StringIO()
    
    tree.export_graphviz(dt, out_file=dot_data,
                         filled=True, rounded=True,
                         special_characters=True)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


# evaluation metrics
def mse(gt, pred):
    return np.mean((gt - pred) ** 2)

def rmse(gt, pred):
    return np.sqrt(mse(gt, pred))

def mae(gt, pred):
    return np.mean(abs(gt - pred))

def r2(gt, pred):
    return 1 - mse(gt, pred) / np.var(gt)

def gini(gt, pred):
    gt_0 = gt - min(gt)
    
    sort_idxs = np.argsort(pred)
    lorenz_pred = np.cumsum(gt_0[sort_idxs]) / sum(gt_0)
    lorenz_gt = np.cumsum(sorted(gt_0)) / sum(gt_0)
    
    middle_line = np.arange(len(gt)) / (len(gt) - 1)
    
    real_gini = sum(middle_line - lorenz_pred)
    ideal_gini = sum(middle_line - lorenz_gt)
    return real_gini / ideal_gini
    
def ks(gt, pred):
    corr_gt = (gt - min(gt)) / (max(gt) - min(gt))
    
    sort_idxs = np.argsort(pred)
    cdf1 = np.cumsum(corr_gt[sort_idxs]) / sum(corr_gt)
    cdf0 = np.cumsum(1-corr_gt[sort_idxs]) / sum(1-corr_gt)
    ks = np.max(np.abs(cdf0 - cdf1))
    return ks

def compute_evaluation_series(gt, pred, name=None):
    if name is None:
        pass
    data = [mse(gt, pred), mae(gt, pred), r2(gt, pred), gini(gt, pred)]
    index = ["MSE", "MAE", "R^2", "Gini"]
    return pd.Series(data, index, name=name)
