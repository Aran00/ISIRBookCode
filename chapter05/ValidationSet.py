__author__ = 'Aran'

import numpy as np
import pandas as pd
from numpy import random as nprand
from sklearn import linear_model
from islrtools import poly
''' Under mac, set where the R library is '''
from sys import platform as _platform
if _platform == 'darwin':
    import os
    os.environ["R_HOME"] = '/Library/Frameworks/R.framework/Versions/3.2/Resources'
import rpy2.robjects as robjects


class ValidationSet:
    def __init__(self, seed=1, use_R_sample=False):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Auto.csv", index_col=0)
        # print self.df.shape, self.df.columns
        self.y_col = 'mpg'
        self.x_cols = ['horsepower']

        if use_R_sample:
            ''' See details in chapter03 - Exec12 '''
            data = robjects.r("""
                set.seed(1)
                x <- sample(392, 196)
            """)
            # Need to substract 1 here
            train = self.df.iloc[np.array(data) - 1].index.values
        else:
            nprand.seed(seed)
            ''' Beware that the first param is not 392 '''
            train = nprand.choice(self.df.index.values, 196, replace=False)

        self.train_set = self.df.ix[train, :]
        test = np.setdiff1d(self.df.index.values, train)
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
    vs = ValidationSet(use_R_sample=True)
    vs.poly_1()
    vs.poly_i(2)
    vs.poly_i(3)
