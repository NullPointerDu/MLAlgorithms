import numpy as np


def lstsq(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
