__author__ = 'ryu'

import pandas as pd
from islrtools import tableplot as tp
from sklearn import neighbors


class KNNClassifier:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
        self.train_set = self.df.ix[self.df.Year < 2005, :]
        self.test_set = self.df.ix[self.df.Year >= 2005, :]
        self.y_col = 'Direction'
        self.x_cols = ['Lag1', 'Lag2']

    def knn_fit_and_pred(self, n_neighbors):
        train_X = self.train_set[self.x_cols].values
        train_y = self.train_set[self.y_col]
        weights = 'uniform'
        #weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(train_X, train_y)
        test_X = self.test_set[self.x_cols].values
        test_y = self.test_set[self.y_col].values
        preds = clf.predict(test_X)
        tp.output_table(preds, test_y)

if __name__ == '__main__':
    knn_example = KNNClassifier()
    # knn_example.knn_fit_and_pred(1)
    knn_example.knn_fit_and_pred(3)