__author__ = 'ryu'

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from examples.plot_separating_hyperplane import plot_svc


class SVCTest:
    def __init__(self, rand_seed=1):
        np.random.seed(1)
        n = 100
        self.X = np.random.standard_normal(size=(2*n, 2))
        self.X[0:100] += 2
        self.X[100:150] -= 2
        self.y = np.array([1]*(3*n/2) + [2]*(n/2))
        print self.y
        ''' train data '''
        train_index = np.random.choice(2*n, n)
        np.zeros((), )
        self.train_X = self.X[train_index]
        self.train_y = self.y[train_index]

        ''' test data '''
        self.test_X = np.random.standard_normal(size=(2*n, 2))
        self.test_y = np.random.choice([-1, 1], 2*n)
        self.test_X[self.test_y == 1] += 1

