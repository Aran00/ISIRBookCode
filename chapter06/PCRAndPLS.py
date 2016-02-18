__author__ = 'Aran'

import numpy as np
import pandas as pd
from sklearn import preprocessing as pp, linear_model as lm
from pandas import Series, DataFrame

class PCRAndPLS:
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
            le = pp.LabelEncoder()
            le.fit(np.unique(self.hitter[col]))
            series = self.hitter.loc[:, col]
            self.hitter.loc[:, col] = Series(le.transform(series.values), index=self.hitter.index)


if __name__ == '__main__':
    pap = PCRAndPLS()
