__author__ = 'Aran'

from islrtools import datasets
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from SplitData import split_data


class Boosting:
    def __init__(self):
        self.target_feature = "medv"
        data = datasets.load_data("Boston", self.target_feature, index_col=0, na_values=['NA'])
        self.data, self.feature_names = data.full, data.feature_names

    def method_test(self):
        train, test = split_data(self.data, self.data.shape[0]/2)
        X, y = train[self.feature_names], train[self.target_feature]
        clf = GradientBoostingRegressor(learning_rate=0.2, n_estimators=5000, max_depth=4)
        clf.fit(X, y)
        y_true = test[self.target_feature]
        y_pred = clf.predict(test[self.feature_names])
        print mean_squared_error(y_true, y_pred)


if __name__ == '__main__':
    bt_test = Boosting()
    bt_test.method_test()
