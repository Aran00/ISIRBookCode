__author__ = 'Aran'

import numpy as np
import pandas as pd
import tableplot as tp
from pandas import DataFrame, Series
from sklearn.lda import LDA


class LDAExamCode:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
        self.train_set = self.df.ix[self.df.Year < 2005, :]
        self.test_set = self.df.ix[self.df.Year >= 2005, :]
        self.y_col = 'Direction'
        self.x_cols = ['Lag1', 'Lag2']

    def lda_fit(self):
        train_X = self.train_set[self.x_cols].values
        train_y = self.train_set[self.y_col]
        fit = LDA(store_covariance=True).fit(train_X, train_y)
        self.print_lda_summary(fit, self.x_cols)
        return fit

    def lda_predict(self, fit, threshold=0.5):
        test_X = self.test_set[self.x_cols].values
        test_y = self.test_set[self.y_col]
        pred_y_probs = fit.predict_proba(test_X)
        pred_y_prob = pred_y_probs[:, 1]
        tp.output_table(pred_y_prob, test_y, zero_one_col_texts=fit.classes_, threshold=threshold)

    def print_lda_summary(self, res, x_col_names):
        print "Prior probabilities of groups:"
        print Series(res.priors_, index=res.classes_)
        print "\n"
        print "Group means:"
        print DataFrame(res.means_, index=res.classes_, columns=x_col_names)
        print "\n"
        print "Coefficients of linear discriminants:"
        print DataFrame(res.scalings_, index=x_col_names, columns=["LD1"])
        '''Who knows the R plot of LDA plot what???'''
        print "\n"


if __name__ == '__main__':
    lec = LDAExamCode()
    #logis.show_correlation()
    fit = lec.lda_fit()
    lec.lda_predict(fit)