import math
import random


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