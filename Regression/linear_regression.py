import numpy as np
from Regression.exceptions import *


class LinearRegression:
    def __init__(self, intercept=True):
        """
        Initialize LinearRegression model.
        :param intercept: If fitting with intercept.
        """
        self.__intercept = intercept
        self.__x = None
        self.__y = None
        self.weight = None

    def fit(self, x, y):
        """
        Fit the model with observations.
        :param x: Training data set observations, rows are individual observations, columns are attributes
        :param y: Training data set results.
        :return: Weights.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self._check_data(x, y)
        if self.__intercept:
            intercept = np.ones((x.shape[0], 1))
            self.__x = np.concatenate((intercept, x), axis=1)
        else:
            self.__x = x
        self.__y = y
        weight = np.linalg.inv(self.__x.T.dot(self.__x)).dot(self.__x.T).dot(self.__y)
        self.weight = weight
        return self.weight

    def _check_data(self, x, y):
        """
        Check the data type
        :param x: Training data set observations, rows are individual observations, columns are attributes.
        :param y: Training data set results.
        """
        # check dtype
        if not np.issubdtype(x.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of x is {}, the data type must be numeric!".format(x.dtype))
        if not np.issubdtype(y.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of y is {}, the data type must be numeric!".format(y.dtype))

        # check shape
        if x.shape[0] != y.shape[0]:
            raise IncorrectMatrixShape("x has shape {} but y has shape {}".format(x.shape, y.shape))

    def estimate(self, x):
        """
        Estimate result with observations.
        :param x: Observations.
        :return: The array contains estimated values.
        """
        if self.weight.shape[0] != (x.shape[1] + 1):
            raise IncorrectMatrixShape("x has shape {} but training data has shape {}".format(x.shape, self.__x.shape))
        if not np.issubdtype(x.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of x is not number!")
        if self.weight is None:
            raise NoDataException("Call fit before estimate!")

        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis=1)
        estimate = x.dot(self.weight)
        return estimate


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from matplotlib import pyplot as plt
    from utils.plot import draw_line

    # generate regression dataset
    X, y = make_regression(n_samples=7, n_features=1, noise=15)
    w = LinearRegression().fit(X, y)
    # plot regression dataset
    ax = plt.gca()
    plt.title("figure 1.1")
    plt.xlabel("x")
    h = plt.ylabel('y')
    h.set_rotation(0)
    s = ax.scatter(X, y)

    # draw regression line
    xmin, xmax = ax.get_xbound()
    l = draw_line([xmin, w[0]+w[1]*xmin], [xmax, w[0]+w[1]*xmax], color='c')


    # draw error
    print(X, y)
    e = None
    for i in range(len(X)):
        x = X[i][0]
        e = draw_line([x, w[0]+w[1]*x], [x, y[i]], color='r')

    ax.legend([s, l, e], ["data points", "regression line", "error"], loc="upper left")
    plt.show()








