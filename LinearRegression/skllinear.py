from sklearn.linear_model import LinearRegression
import numpy as np


if __name__ == "__main__":
    x = np.array([[12, 2], [24, 5], [41, 9]])
    y = np.array([[1, 2], [2, 6], [4, 10]])
    reg = LinearRegression(fit_intercept=True).fit(x, y)
    print(reg.intercept_)
    print(reg.coef_)
    test_x = np.array([[12, 2], [24, 5], [41, 100]])
    print(reg.predict(test_x))