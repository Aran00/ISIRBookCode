__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from examples.plot_voting_decision_regions import plot_contour


class SVMTest:
    def __init__(self, rand_seed=1):
        np.random.seed(rand_seed)
        n = 100
        self.X = np.random.standard_normal(size=(2*n, 2))
        self.X[0:100] += 2
        self.X[100:150] -= 2
        self.y = np.array([1]*(3*n/2) + [2]*(n/2))
        # print self.y
        ''' train data '''
        train_index_raw = np.random.choice(2*n, n, replace=False)
        train_index = np.zeros(2*n, dtype=np.bool)
        train_index[train_index_raw] = 1

        self.train_X = self.X[train_index]
        self.train_y = self.y[train_index]
        print len(self.train_y)

        ''' test data '''
        self.test_X = self.X[~train_index]
        self.test_y = self.y[~train_index]
        self.test_X[self.test_y == 1] += 1
        print len(self.test_y)

    def scatter_x(self, clf):
        """ cmap is used when c is integer """
        # c = map(lambda z: 'g' if z == 1 else 'r', self.y)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Paired)
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                    s=80, facecolors='none')
        plt.show()

    ''' When we adjust the params once again,
        we can see that gamma controls the range of inner class.
        If gamma is too large, the inner class range would be one point at last,
        as all kernels value would limit to 0.
        In other word, large gamma increases the model complexity.
    '''
    def train_svm_and_plot(self):
        clf = SVC(kernel="rbf", C=1, gamma=1)
        clf.fit(self.train_X, self.train_y)
        # print clf.dual_coef_, "\n", clf.intercept_, "\n", clf.support_, "\n", clf.support_vectors_
        cs = plot_contour(clf, self.train_X)
        # plt.clabel(cs, inline=1, fontsize=10)
        self.scatter_x(clf)
        return clf

    def choose_params(self):
        parameters = {
            'C': np.logspace(-1, 3, 5),
            'gamma': [0.5, 1, 2, 3, 4]
        }
        svr = SVC(kernel="rbf")
        '''
        cv: divide data into train and validation set.
            So we still need another test set to see its test performance.
        '''
        search = GridSearchCV(svr, parameters, cv=10)
        search.fit(self.train_X, self.train_y)
        print search.best_params_
        return search.best_estimator_

    def get_test_result(self, clf):
        pred_y = clf.predict(self.test_X)
        cm = confusion_matrix(self.test_y, pred_y)
        cm = np.matrix(cm)
        print cm, 1.0 * np.matrix.trace(cm)/np.matrix.sum(cm)


if __name__ == '__main__':
    st = SVMTest()
    best_model = st.train_svm_and_plot()
    # best_model = st.choose_params()
    st.get_test_result(best_model)
