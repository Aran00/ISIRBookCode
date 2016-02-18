__author__ = 'Aran'

import numpy as np
import pandas as pd
from sklearn import preprocessing as pp, linear_model
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from pandas import Series, DataFrame


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
        self.alphas = np.power(10, np.linspace(10, -2, 100))

    def transform_label(self):
        trans_cols = ["League", "Division", "NewLeague"]
        for col in trans_cols:
            le = pp.LabelEncoder()
            le.fit(np.unique(self.hitter[col]))
            series = self.hitter.loc[:, col]
            self.hitter.loc[:, col] = Series(le.transform(series.values), index=self.hitter.index)

    def penalty_test(self, ridge=True):
        ''' See here: http://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons '''
        result = DataFrame(index=np.arange(len(self.alphas)), columns=self.x_cols)
        for i, alpha in enumerate(self.alphas):
            # alpha = 111497.57/2.0  #11498
            clf = linear_model.Ridge(alpha=alpha) if ridge else linear_model.Lasso(alpha=alpha)
            # print self.X.shape, self.y.shape
            # print self.X.iloc[0]
            clf.fit(self.X, self.y)
            result.ix[i, :] = clf.coef_
            # print clf.intercept_, clf.coef_.shape, "\n", zip(self.x_cols, clf.coef_)
            # print (clf.intercept_ + np.sum(clf.coef_)), clf.predict(np.ones((1, 19)))
        print result.ix[[0, 20, 40, 60, 80, 99], :]

    def ridge_standarize_test(self):
        ''' See here: http://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons '''
        alpha = 11498
        clf = linear_model.Ridge(alpha=alpha)
        clf.fit(self.X, self.y)
        print clf.coef_

        clf = linear_model.Ridge(alpha=alpha)
        clf.fit(pp.scale(self.X), pp.scale(self.y))
        print clf.coef_

        clf = linear_model.Ridge(alpha=alpha, normalize=True)
        clf.fit(self.X, self.y)
        print clf.coef_
        # print clf.intercept_, clf.coef_.shape, "\n", zip(self.x_cols, clf.coef_)
        # print (clf.intercept_ + np.sum(clf.coef_)), clf.predict(np.ones((1, 19)))

    def cv_test(self, ridge=True):
        ''' When cv is 5, 10 or 20, the alpha_ result is totally different '''
        clf = linear_model.RidgeCV(alphas=self.alphas, cv=10) if ridge else \
            linear_model.LassoCV(alphas=self.alphas, cv=10, max_iter=10000)
        clf.fit(self.X, self.y)
        print clf.alpha_, "\n", zip(self.x_cols, clf.coef_)
        print sum(clf.coef_ != 0)

    def grid_search_test(self):
        clf = linear_model.Lasso(max_iter=10000)
        grid_search_params = {
            'alpha': self.alphas
        }
        ''' The test shows that ridge use r2 and lasso uses mean squared error as score? Interesting... '''
        gs = GridSearchCV(clf, grid_search_params, cv=10, scoring="mean_squared_error") #, scoring="r2")
        gs.fit(self.X, self.y)
        print gs.grid_scores_, '\n', max(gs.grid_scores_, key=lambda x: x[1])


if __name__ == '__main__':
    rl = RidgeAndLasso()
    # rl.penalty_test(False)
    rl.cv_test(False)
    rl.grid_search_test()
