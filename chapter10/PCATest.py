__author__ = 'Aran'

import numpy as np
from islrtools import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from pandas import DataFrame


class PCATest:
    def __init__(self):
        data = datasets.load_data("USArrests", None, index_col=0, na_values=['NA'])
        self.X, self.feature_names = data.full, data.feature_names

    def test_mean_and_std(self):
        mean_and_std = DataFrame(np.zeros((2, len(self.feature_names))),
                                 index=['mean', 'std'], columns=self.feature_names)
        for column in self.feature_names:
            mean_and_std[column] = [np.mean(self.X[column]), np.std(self.X[column])]
        print mean_and_std

    def test_pca(self):
        X = scale(self.X)
        pca = PCA(n_components=4)
        clf = pca.fit(X)
        X_r = clf.transform(X)
        print X_r
        print clf.explained_variance_ratio_


if __name__ == '__main__':
    pca_test = PCATest()
    # pca_test.test_mean_and_std()
    pca_test.test_pca()
