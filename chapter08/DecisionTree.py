__author__ = 'Aran'

from islrtools import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
from numpy import random
from time import time
from operator import itemgetter
import numpy as np
import pydot


'''
The decision tree in scikit-learn doesn't support pruning
So we only can set min sample number in leaf or maximum depth(I think these 2 have similar effect) to avoid overfit
'''
class DecisionTree:
    def __init__(self):
        self.target_feature = "Sales"
        data = datasets.load_data("Carseats", self.target_feature, index_col=0, na_values=['NA'])
        self.hitter, self.feature_names = data.full, data.feature_names
        self.hitter[self.target_feature] = self.hitter[self.target_feature].map(lambda x: 1 if x > 8 else 0)

    def split_data(self):
        np.random.seed(1)
        train_index = random.choice(self.hitter.index, 200, replace=False)
        # print train_index
        train_cond = self.hitter.index.isin(train_index)
        return self.hitter.ix[train_cond], self.hitter.ix[~train_cond]

    ''' Use cv to choose parameters '''
    def choose_params(self):
        train, test = self.split_data()
        X, y = train[self.feature_names], train[self.target_feature]
        dt = DecisionTreeClassifier()
        param_dist = {
            # "criterion": ["gini", "entropy"],
            "max_features": np.arange(1, 9),
            "max_depth": np.arange(2, 16)
        }
        n_iter_search = 20
        random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=n_iter_search)
        for i in xrange(10):
            np.random.seed(i)
            start = time()
            random_search.fit(X, y)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            DecisionTree.report(random_search.grid_scores_, n_top=1)
            top_score = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[0]
            print "Randomized search choose best:", top_score.parameters

        # run grid search
        np.random.seed(1)
        grid_search = GridSearchCV(dt, param_grid=param_dist)
        start = time()
        grid_search.fit(X, y)
        print("\n\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.grid_scores_)))
        DecisionTree.report(grid_search.grid_scores_, n_top=3)


    ''' An overfit tree '''
    def test_decision_tree(self):
        dt = DecisionTreeClassifier()
        X = self.hitter[self.feature_names]
        y = self.hitter[self.target_feature]
        dt.fit(X, y)
        preds = dt.predict(X)
        print dt, "\n", dt.n_features_
        print (y == preds).mean()
        self.print_decision_tree(dt)
        self.print_detail(dt)

    def print_decision_tree(self, dt):
        clf = tree.DecisionTreeClassifier()
        iris = load_iris()
        dt = clf.fit(iris.data, iris.target)
        str_buffer = StringIO()
        tree.export_graphviz(dt, out_file=str_buffer)
        graph = pydot.graph_from_dot_data(str_buffer.getvalue())
        graph.write_pdf("myfile.pdf")

    def print_detail(self, dt):
        print dt.classes_, dt.feature_importances_, dt.max_features_, dt.n_features_, dt.tree_.node_count

    @staticmethod
    def report(grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

if __name__ == '__main__':
    dt_test = DecisionTree()
    # dt_test.test_decision_tree()
    dt_test.choose_params()
