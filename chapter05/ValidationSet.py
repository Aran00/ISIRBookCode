__author__ = 'Aran'

import numpy as np
import pandas as pd
from numpy import random as nprand
from sklearn import linear_model
from islrtools import poly


class ValidationSet:
    def __init__(self, seed=1):
        auto = pd.read_csv("../../ISIRExerciseCode/dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()
        # print self.df.shape, self.df.columns
        self.y_col = 'mpg'
        self.x_cols = ['horsepower']
        nprand.seed(seed)
        ''' Beware that the first param is not 392 '''
        train = nprand.choice(self.df.index.values, 196, replace=False)
        self.train_set = self.df.ix[train, :]
        test = np.setdiff1d(self.df.index.values, train)
        print len(test)
        self.test_set = self.df.ix[test, :]

    def poly_1(self):
        clf = linear_model.LinearRegression()
        clf.fit(self.train_set[self.x_cols], self.train_set[self.y_col])
        print np.mean(np.square(self.test_set[self.y_col] - clf.predict(self.test_set[self.x_cols])))

    def poly_i(self, degree):
        clf = linear_model.LinearRegression()
        Z, norm2, alpha = poly.ortho_poly_fit(self.train_set[self.x_cols], degree)
        clf.fit(Z, self.train_set[self.y_col])
        predict_values = clf.predict(poly.ortho_poly_predict(self.test_set[self.x_cols[0]], alpha, norm2, degree))
        print np.mean(np.square(self.test_set[self.y_col] - predict_values))

if __name__ == '__main__':
    vs = ValidationSet(2)
    vs.poly_1()
    vs.poly_i(2)
    vs.poly_i(3)
