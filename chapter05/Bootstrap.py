__author__ = 'Aran'

import numpy as np
import pandas as pd
from islrtools import poly
from islrtools import bootstrap as bsp
from numpy import random as nprand
from sklearn import linear_model


class Bootstrap:
    def __init__(self):
        self.portfolio = pd.read_csv("../../ISIRExerciseCode/dataset/Portfolio.csv")
        self.auto = pd.read_csv("../../ISIRExerciseCode/dataset/Auto.csv")

    @staticmethod
    def calculate_alpha(data, index):
        x = data.ix[index]['X']
        y = data.ix[index]['Y']
        cov_xy = np.cov(x, y)[0][1]
        return (np.var(y, ddof=1) - cov_xy)/(np.var(x, ddof=1) + np.var(y, ddof=1) - 2*cov_xy)

    @staticmethod
    def linear_modal_coef(degree, orthogonality=True):
        def func(data, index):
            X = data.ix[index][['horsepower']]
            y = data.ix[index]['mpg']
            clf = linear_model.LinearRegression()
            if degree > 1:
                if orthogonality:
                    Z, norm2, alpha = poly.ortho_poly_fit(X, degree)
                    X = Z[:, 1:]
                else:
                    Z = np.zeros((X.shape[0], degree))
                    for i in xrange(degree):
                        Z[:, i] = np.power(X.ix[:, 0], i+1)
                    X = Z
            clf.fit(X, y)
            result = np.append(clf.intercept_, clf.coef_)
            return result
        return func

    @staticmethod
    def poly_model_coef(data, index):
        X = data.ix[index][['horsepower']]
        y = data.ix[index]['mpg']
        clf = linear_model.LinearRegression()
        clf.fit(X, y)
        return [clf.intercept_, clf.coef_[0]]


if __name__ == '__main__':
    bp = Bootstrap()
    nprand.seed(0)
    '''
    print Bootstrap.calculate_alpha(bp.portfolio, range(len(bp.portfolio)))
    print bsp.boot(bp.portfolio, Bootstrap.calculate_alpha, 1000)
    '''
    real_func = Bootstrap.linear_modal_coef(2, False)
    print real_func(bp.auto, range(len(bp.auto)))
    print bsp.boot(bp.auto, real_func, 1000)
