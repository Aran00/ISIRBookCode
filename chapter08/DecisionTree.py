__author__ = 'Aran'

from islrtools import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import tree
from sklearn.externals.six import StringIO
from numpy import random
from time import time
from operator import itemgetter
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pydot


'''
The decision tree in scikit-learn doesn't support pruning
So we only can set min sample number in leaf or maximum depth(I think these 2 have similar effect) to avoid overfit
'''
class DecisionTree:
    def __init__(self, classification=True):
        self.classification = classification
        self.target_feature = "Sales" if classification else "medv"
        data = datasets.load_data("Carseats" if classification else "Boston",
                                  self.target_feature, index_col=0, na_values=['NA'])
        self.data, self.feature_names = data.full, data.feature_names
        self.data[self.target_feature] = self.data[self.target_feature].map(lambda x: 1 if x > 8 else 0)

    def split_data(self):
        np.random.seed(1)
        train_index = random.choice(self.data.index, 200, replace=False)
        # print train_index
        train_cond = self.data.index.isin(train_index)
        return self.data.ix[train_cond], self.data.ix[~train_cond]

    ''' Use cv to choose parameters '''
    def choose_params(self, random_search=False):
        train, test = self.split_data()
        X, y = train[self.feature_names], train[self.target_feature]
        dt = DecisionTreeClassifier() if self.classification else DecisionTreeRegressor()
        param_dist = {
            # "criterion": ["gini", "entropy"],
            "max_features": np.arange(1, len(self.feature_names) + 1),
            "max_leaf_nodes": np.arange(2, 20)
        }

        np.random.seed(1)
        search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=20) \
            if random_search else GridSearchCV(dt, param_grid=param_dist)
        start = time()
        search.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), 20))
        DecisionTree.report(search.grid_scores_, n_top=1)
        top_score = sorted(search.grid_scores_, key=itemgetter(1), reverse=True)[0]
        print "Randomized search choose best:", top_score.parameters
        best_param = search.best_params_
        print best_param
        best_estimator = search.best_estimator_

        # Print test data confusion matrix
        y_true = test[self.target_feature]
        y_pred = best_estimator.predict(test[self.feature_names])
        if self.classification:
            cm = confusion_matrix(y_true, y_pred)
            print cm
            print classification_report(y_true, y_pred)
        else:
            print "The MSE is:", np.mean(np.square(y_true - y_pred))
        self.output_final_tree(**best_param)

    ''' Use the params from train set cv and all data to train a new tree '''
    def output_final_tree(self, **param):
        init_func = DecisionTreeClassifier if self.classification else DecisionTreeRegressor
        dt = init_func(**param)
        X = self.data[self.feature_names]
        y = self.data[self.target_feature]
        dt.fit(X, y)
        preds = dt.predict(X)
        if self.classification:
            print "The final train accuracy is:", (y == preds).mean()   # test accuracy
        else:
            print "The final train MSE is:", np.mean(np.square(y - preds))
        self.print_decision_tree(dt)
        self.print_detail(dt)

    def print_decision_tree(self, dt):
        str_buffer = StringIO()
        tree.export_graphviz(dt, out_file=str_buffer, feature_names=self.feature_names)
        graph = pydot.graph_from_dot_data(str_buffer.getvalue())
        graph.write_pdf("output/%s.pdf" % self.target_feature)

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


if __name__ == '__main__':
    dt_test = DecisionTree()
    dt_test.choose_params(True)
