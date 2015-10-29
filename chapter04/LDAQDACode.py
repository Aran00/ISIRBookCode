__author__ = 'Aran'

import numpy as np
import pandas as pd
from islrtools import tableplot as tp
from pandas import DataFrame, Series
from sklearn.lda import LDA
from sklearn.qda import QDA


class LDAQDACode:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
        self.train_set = self.df.ix[self.df.Year < 2005, :]
        self.test_set = self.df.ix[self.df.Year >= 2005, :]
        self.y_col = 'Direction'
        self.x_cols = ['Lag1', 'Lag2']

    def lqda_fit(self, lda=True):
        train_X = self.train_set[self.x_cols].values
        train_y = self.train_set[self.y_col]
        if lda is True:
            fit_res = LDA(store_covariance=True).fit(train_X, train_y)
        else:
            fit_res = QDA().fit(train_X, train_y, store_covariances=True)
        self.print_summary(fit_res, self.x_cols, lda)
        return fit_res

    def lda_predict(self, fit_res, threshold=0.5):
        test_X = self.test_set[self.x_cols].values
        test_y = self.test_set[self.y_col].values
        if threshold == 0.5:
            pred_y = fit_res.predict(test_X)
        else:
            pred_y_probs = fit_res.predict_proba(test_X)
            pred_y = np.array([fit_res.classes_[0] if pred_y_probs[i, 0] > threshold else fit_res.classes_[1]
                               for i in xrange(pred_y_probs.shape[0])])
        tp.output_table(pred_y, test_y)

    def print_summary(self, res, x_col_names, lda=True):
        print "Prior probabilities of groups:"
        print Series(res.priors_, index=res.classes_)
        print "\n"
        print "Group means:"
        print DataFrame(res.means_, index=res.classes_, columns=x_col_names)
        print "\n"
        if lda is True:
            print "Covariance is:"
            print res.covariance_
            print "\n"
            print "Coefficients of linear discriminants:"
            print DataFrame(res.scalings_, index=x_col_names, columns=["LD1"])
            '''Who knows the R plot of LDA plot what???'''
        else:
            print "Covariance is:"
            print res.covariances_
        print "\n"

if __name__ == '__main__':
    lec = LDAQDACode()
    fit_result = lec.lqda_fit(True)
    lec.lda_predict(fit_result, threshold=0.5)
