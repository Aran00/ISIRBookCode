__author__ = 'Aran'

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
from pandas import Series

class RidgeAndLasso:
    def __init__(self):
        hitter = pd.read_csv("../../ISIRExerciseCode/dataset/Hitter.csv", index_col=0, na_values=['NA'])
        self.hitter = hitter.dropna()
        self.transform_label()
        self.y_col = 'Salary'
        self.y = self.hitter[self.y_col]
        self.x_cols = self.hitter.columns.tolist()
        self.x_cols.remove(self.y_col)
        self.X = self.hitter.ix[:, self.x_cols]

    def transform_label(self):
        trans_cols = ["League", "Division", "NewLeague"]
        for col in trans_cols:
            le = preprocessing.LabelEncoder()
            le.fit(np.unique(self.hitter[col]))
            series = self.hitter.loc[:, col]
            self.hitter.loc[:, col] = Series(le.transform(series.values), index=self.hitter.index)

    def ridge_test(self):
        ''' See here: http://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons '''
        alphas = np.power(10, np.linspace(10, -2, 100))
        # for alpha in alphas:
        alpha = 11497.57/2.0  #11498
        clf = linear_model.Ridge(alpha=alpha)
        print self.X.shape, self.y.shape
        print self.X.iloc[0]
        clf.fit(self.X, self.y)
        print clf.intercept_, clf.coef_.shape, "\n", zip(self.x_cols, clf.coef_)

    def ridge_cv_test(self):
        alphas = np.power(10, np.linspace(10, -2, 100))
        clf = linear_model.RidgeCV(alphas=alphas, cv=5)
        clf.fit(self.X, self.y)
        print clf.alpha_, clf.coef_.shape, "\n", zip(self.x_cols, clf.coef_)


if __name__ == '__main__':
    rl = RidgeAndLasso()
    rl.ridge_test()
