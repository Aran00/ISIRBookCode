__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from examples.plot_separating_hyperplane import plot_svc


class SVCTest:
    def __init__(self, rand_seed):
        np.random.seed(rand_seed)
        # X = np.random.randn(20, 2)
        n = 10
        ''' train data '''
        self.X = np.random.standard_normal(size=(2*n, 2))
        # y = np.hstack((-np.ones(n), np.ones(n)))
        self.y = np.array([-1]*n + [1]*n)
        self.X[self.y == 1] += 1
        ''' test data '''
        self.test_X = np.random.standard_normal(size=(2*n, 2))
        self.test_y = np.random.choice([-1, 1], 2*n)
        self.test_X[self.test_y == 1] += 1

    def scatter_X(self):
        """ cmap is used when c is integer """
        # c = map(lambda z: 'g' if z == 1 else 'r', self.y)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Paired)
        plt.show()

    def svc_init(self, cost):
        svc = SVC(kernel="linear", C=cost)
        svc.fit(self.X, self.y)
        self.print_svc_detail(svc)
        return svc

    def svc_cv_test(self):
        # parameters = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}
        parameters = {'C': np.logspace(-3, 2, 6)}
        svr = SVC(kernel="linear")
        search = GridSearchCV(svr, parameters, cv=10)
        search.fit(self.X, self.y)
        print search.best_params_, "\n", search.best_score_
        return search.best_estimator_

    def test_set_verify(self, clf):
        y_pred = clf.predict(self.test_X)
        print y_pred, "\n", self.test_y
        cm = confusion_matrix(self.test_y, y_pred)
        print cm
        print classification_report(self.test_y, y_pred)

    def further_change_data(self):
        self.X[self.y == 1] += 1
        self.scatter_X()

    @staticmethod
    def print_svc_detail(clf):
        print clf.dual_coef_, "\n", clf.intercept_, "\n", clf.support_, "\n", clf.support_vectors_


if __name__ == '__main__':
    st = SVCTest(2)
    st.further_change_data()
    # best_model = st.svc_cv_test()
    best_model = st.svc_init(cost=1e5)
    st.test_set_verify(best_model)
    plot_svc(best_model, st.X, st.y)
