__author__ = 'Aran'

import numpy as np
import pandas as pd
import cv_methods as cm
from islrtools import poly
from sklearn import linear_model
from sklearn import cross_validation

class CrossValidation:
    def __init__(self):
        ''' What's the data management method of scilearn-kit package? '''
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Auto.csv")
        # print self.df.shape, self.df.columns
        self.y_col = 'mpg'
        self.x_cols = ['horsepower']
        self.data_len = self.df.shape[0]

    def linear_modal_coef(self):
        clf = linear_model.LinearRegression()
        clf.fit(self.df[self.x_cols], self.df[self.y_col])
        print clf.intercept_, clf.coef_[0]

    def cross_validation(self, cv_num):
        clf = linear_model.LinearRegression()
        scores = cross_validation.cross_val_score(clf, self.df[self.x_cols], self.df[self.y_col],
                                                  scoring="mean_squared_error", cv=cv_num)
        print len(scores), - np.mean(scores), np.std(scores)

    def cross_poly_i(self, degree, cv):
        clf = linear_model.LinearRegression()
        y = self.df[self.y_col]
        if degree == 1:
            X = self.df[self.x_cols]
        else:
            X, norm2, alpha = poly.ortho_poly_fit(self.df[self.x_cols], degree)
        ''' These scores should be equal with R for the LOOCV, but not equal for the case folds != data_len '''
        scores = cross_validation.cross_val_score(clf, X, y, scoring="mean_squared_error", cv=cv)
        print degree, len(scores), -np.mean(scores), np.std(scores)


if __name__ == '__main__':
    cv_test = CrossValidation()
    '''
    for i in xrange(5):
        cv_test.cross_poly_i(degree=i+1, cv=cv_test.data_len)
    '''
    for i in xrange(10):
        cv_test.cross_poly_i(degree=i+1, cv=cm.kfold_iterator(cv_test.data_len, 10, True, 0))
