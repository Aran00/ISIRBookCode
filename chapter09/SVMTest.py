__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from examples.plot_separating_hyperplane import plot_svc


class SVMTest:
    def __init__(self, rand_seed=1):
        np.random.seed(1)
        n = 100
        self.X = np.random.standard_normal(size=(2*n, 2))
        self.X[0:100] += 2
        self.X[100:150] -= 2
        self.y = np.array([1]*(3*n/2) + [2]*(n/2))
        print self.y
        ''' train data '''
        train_index_raw = np.random.choice(2*n, n)
        train_index = np.zeros(2*n, dtype=np.bool)
        train_index[train_index_raw] = 1

        self.train_X = self.X[train_index]
        self.train_y = self.y[train_index]

        ''' test data '''
        self.test_X = self.X[~train_index]
        self.test_y = self.y[~train_index]
        self.test_X[self.test_y == 1] += 1

    def scatter_x(self):
        """ cmap is used when c is integer """
        # c = map(lambda z: 'g' if z == 1 else 'r', self.y)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Paired)
        plt.show()

if __name__ == '__main__':
    st = SVMTest()
    st.scatter_x()