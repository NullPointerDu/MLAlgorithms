import numpy as np
from LinearRegression.exceptions import *


class LinearRegression:
    def __init__(self):
        self.__x = None
        self.__y = None
        self.__w = None

    def fit(self, x, y):
        """
        Fit the model with observations.
        :param x: Training data set observations, rows are individual observations, columns are attributes
        :param y: Training data set results.
        :return: Weights.
        """
        self._check_data(x, y)
        intercept = np.ones((x.shape[0], 1))
        self.__x = np.concatenate((intercept, x), axis=1)
        self.__y = y
        weight = np.linalg.inv(self.__x.T.dot(self.__x)).dot(self.__x.T).dot(self.__y)
        self.__w = weight
        return self.__w

    def _check_data(self, x, y):
        """
        Check the data type
        :param x: Training data set observations, rows are individual observations, columns are attributes.
        :param y: Training data set results.
        """
        # check type
        if not isinstance(x, np.ndarray):
            raise IncorrectObjectException("x is not an instance of np.ndarray")
        if not isinstance(y, np.ndarray):
            raise IncorrectObjectException("y is not an instance of np.ndarray")

        # check dtype
        if not np.issubdtype(x.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of x is not number!")
        if not np.issubdtype(y.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of y is not number!")

        # check shape
        if x.shape[0] != y.shape[0]:
            raise IncorrectMatrixShape("x has shape {} but y has shape {}".format(x.shape, y.shape))

    def estimate(self, x):
        """
        Estimate result with observations.
        :param x: Observations.
        :return: The array contains estimated values.
        """
        if self.__w.shape[0] != (x.shape[1] + 1):
            raise IncorrectMatrixShape("x has shape {} but training data has shape {}".format(x.shape, self.__x.shape))
        if not np.issubdtype(x.dtype, np.number):
            raise DtypeIsNotNumeric("The dtype of x is not number!")
        if self.__w is None:
            raise NoDataException("Call fit before estimate!")

        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis=1)
        estimate = x.dot(self.__w)
        return estimate


if __name__ == "__main__":
    x = np.array([[12, 2], [24, 5], [41, 9]])
    y = np.array([[1, 2], [2, 6], [4, 10]])
    reg = LinearRegression()
    print(reg.fit(x, y))
    test_x = np.array([[12, 2], [24, 5], [41, 7]])
    y = reg.estimate(test_x)
    print(y)
