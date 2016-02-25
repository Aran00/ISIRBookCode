__author__ = 'Aran'

from islrtools import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
from sklearn.datasets import load_iris


class DecisionTree:
    def __init__(self):
        hitter = datasets.load_data("Carseats", "Sales", index_col=0, na_values=['NA'])
        self.X, y = hitter.data, hitter.target
        self.y = y.map(lambda x: 1 if x > 8 else 0)

    def test_decision_tree(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.X, self.y)
        preds = dt.predict(self.X)
        print dt, "\n", dt.n_features_
        print (self.y == preds).mean()
        self.print_decision_tree(dt)

    def print_decision_tree(self, dt):
        clf = tree.DecisionTreeClassifier()
        iris = load_iris()
        dt = clf.fit(iris.data, iris.target)
        # str_buffer = StringIO()
        tree.export_graphviz(dt, out_file="tree.dot")
        '''
        graph = pydot.graph_from_dot_data(str_buffer.getvalue())
        graph.write_pdf("myfile.pdf")
        '''

if __name__ == '__main__':
    dt_test = DecisionTree()
    dt_test.test_decision_tree()
