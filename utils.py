import math
import random
import numpy as np


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


# solve linear models
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
    
    # denormalize
    y_hat = mean_y + sd_y * norm_y_hat
    return y_hat, norm_w_hat


# evaluation metrics
def mse(gt, pred):
    return np.sum((gt - pred) ** 2) / len(gt)

def mae(gt, pred):
    return np.sum(np.abs(gt - pred)) / len(gt)

def r2(gt, pred):
    return 1 - mse(gt, pred) / np.var(gt)