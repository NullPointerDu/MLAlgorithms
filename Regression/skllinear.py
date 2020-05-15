from sklearn.linear_model import LinearRegression
import numpy as np


if __name__ == "__main__":
    x = np.array([[12, 2], [24, 5], [41, 9]])
    y = np.array([[1, 2], [2, 6], [4, 10]])
    import time
    start = time.time()
    LinearRegression(fit_intercept=True).fit(x, y)
    print(time.time() - start)
