__author__ = 'Aran'

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import Series
from islrtools import subset
from sklearn import preprocessing, linear_model, cross_validation
from sys import platform as _platform
if _platform == 'darwin':
    import os
    os.environ["R_HOME"] = '/Library/Frameworks/R.framework/Versions/3.2/Resources'
import rpy2.robjects as robjects


class SubsetSelection:
    def __init__(self):
        hitter = pd.read_csv("../../ISIRExerciseCode/dataset/Hitter.csv", index_col=0, na_values=['NA'])
        self.hitter = hitter.dropna()
        self.transform_label()
        train_index, test_index = self.train_index()
        self.train_set = self.hitter.iloc[train_index]
        self.test_set = self.hitter.iloc[test_index]

    def transform_label(self):
        trans_cols = ["League", "Division", "NewLeague"]
        for col in trans_cols:
            le = preprocessing.LabelEncoder()
            le.fit(np.unique(self.hitter[col]))
            series = self.hitter.loc[:, col]
            self.hitter.loc[:, col] = Series(le.transform(series.values), index=self.hitter.index)

    def train_index(self):
        ''' See details in chapter03 - Exec12 '''
        data = robjects.r("""
            set.seed(1)
            train=sample(c(1, 0), 263, rep=1)
        """)
        # Need to substract 1 here
        train = np.array(data, dtype=int)
        train_index = []
        test_index = []
        for i in xrange(len(train)):
            if train[i] == 1:
                train_index.append(i)
            else:
                test_index.append(i)
        return train_index, test_index

    def output_best_subsets(self):
        subset_seq, stat = subset.regsubsets("Salary~.", self.hitter, 19, "backward")
        self.choose_best_by_stat(stat)

    def choose_best_by_stat(self, stat_data):
        stat_bic = stat_data.map(lambda x: x.bic)
        idx_min = stat_bic.idxmin()
        print idx_min
        print stat_data[7].params

    def test_set_best(self):
        subset_seq, stat = subset.regsubsets("Salary~.", self.train_set, 19, "forward")
        test_mse = Series(np.zeros(len(stat)), index=stat.index)
        for idx in subset_seq.index:
            x_cols = self.get_used_x_cols(subset_seq, idx)
            ols_result = stat[idx]
            test_pred_val = ols_result.predict(sm.add_constant(self.test_set.ix[:, x_cols]))
            test_mse[idx] = np.sum(np.square(test_pred_val - self.test_set.ix[:, "Salary"].values))/self.test_set.shape[0]
        print test_mse, "\n", test_mse.idxmin(), np.std(test_mse)

    def cross_validation_best(self):
        subset_seq, stat = subset.regsubsets("Salary~.", self.train_set, 19, "forward")
        cv_mse = Series(np.zeros(len(stat)), index=stat.index)
        for idx in subset_seq.index:
            x_cols = self.get_used_x_cols(subset_seq, idx)
            clf = linear_model.LinearRegression()
            scores = cross_validation.cross_val_score(clf, self.hitter[x_cols], self.hitter['Salary'], scoring="mean_squared_error", cv=10)
            cv_mse[idx] = -np.mean(scores)
        print cv_mse, "\n", cv_mse.idxmin(), np.std(cv_mse)

    @staticmethod
    def get_used_x_cols(subset_seq, i):
        x_cols = []
        for col_name in subset_seq.columns:
            if subset_seq.ix[i][col_name]:
                x_cols.append(col_name)
        return x_cols

if __name__ == '__main__':
    ss = SubsetSelection()
    # ss.test_set_best()
    # ss.output_best_subsets()
    ss.cross_validation_best()
