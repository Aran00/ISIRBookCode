__author__ = 'Aran'

from islrtools import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
import pydot


class DecisionTree:
    def __init__(self):
        hitter = datasets.load_data("Carseats", "Sales", index_col=0, na_values=['NA'])
        self.X, y = hitter.data, hitter.target
        self.y = y.map(lambda x: 1 if x > 8 else 0)
        # print self.X.iloc[0]

    def test_carseat_dt(self):
        dt = DecisionTreeClassifier(max_features=7, max_depth=4)
        dt.fit(self.X, self.y)
        preds = dt.predict(self.X)
        print dt, "\n", dt.n_features_
        print (self.y == preds).mean()
        self.print_detail(dt)
        self.print_decision_tree(dt, self.X.columns, "carseat")

    def test_iris_dt(self):
        iris = load_iris()
        clf = tree.DecisionTreeClassifier()
        clf.fit(iris.data, iris.target)
        print iris.data.shape, iris.feature_names
        self.print_detail(clf)
        self.print_decision_tree(clf, iris.feature_names, "iris")

    def print_decision_tree(self, dt, feature_names, file_name):
        str_buffer = StringIO()
        tree.export_graphviz(dt, feature_names=feature_names, out_file=str_buffer)
        graph = pydot.graph_from_dot_data(str_buffer.getvalue())
        graph.write_pdf("%s.pdf" % file_name)

    def print_detail(self, dt):
        print dt.classes_, dt.feature_importances_, dt.max_features_, dt.n_features_, dt.tree_.node_count


if __name__ == '__main__':
    dt_test = DecisionTree()
    dt_test.test_carseat_dt()
    # dt_test.test_iris_dt()