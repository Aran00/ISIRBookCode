__author__ = 'Aran'

import matplotlib.pyplot as plt
from numpy import random as nprand
from sklearn.cluster import KMeans


class KMeanTests:
    N = 50

    def __init__(self):
        n = KMeanTests.N
        nprand.seed(2)
        self.X = nprand.standard_normal((n, 2))
        print self.X.shape
        self.X[0:n/2, 0] += 3
        self.X[0:n/2, 1] -= 4

    def test_kmeans(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(self.X)
        print kmeans.labels_
        print kmeans.inertia_
        plt.scatter(self.X[:, 0], self.X[:, 1], c=kmeans.labels_)
        plt.show()


if __name__ == '__main__':
    kmeans_test = KMeanTests()
    kmeans_test.test_kmeans(3)
