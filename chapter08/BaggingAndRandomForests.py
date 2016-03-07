__author__ = 'Aran'

import numpy as np
from islrtools import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from SplitData import split_data, report


class BaggingAndRandomForests:
    def __init__(self):
        self.target_feature = "medv"
        data = datasets.load_data("Boston", self.target_feature, index_col=0, na_values=['NA'])
        self.data, self.feature_names = data.full, data.feature_names

    def method_test(self, n_estimators=25, bagging=True, random_seed=0):
        train, test = split_data(self.data, 3*self.data.shape[0]/4, rand_seed=random_seed)
        X, y = train[self.feature_names], train[self.target_feature]
        rf = RandomForestRegressor(n_estimators=n_estimators, max_features=len(self.feature_names) if bagging else "sqrt")
        rf.fit(X, y)
        # self.print_detail(rf)
        y_true = test[self.target_feature]
        y_pred = rf.predict(test[self.feature_names])
        test_mse = np.mean(np.square(y_true - y_pred))
        print "The %s test MSE is:" % ("bagging" if bagging else "Random forest"), test_mse
        return test_mse

    def print_detail(self, rf):
        print zip(self.feature_names, rf.feature_importances_), rf.n_features_

    def cv_judge(self, seed=0):
        train, test = split_data(self.data, 3*self.data.shape[0]/4, rand_seed=0)
        X, y = train[self.feature_names], train[self.target_feature]
        rf = RandomForestRegressor(n_estimators=50)
        param_dist = {
            "max_features": np.arange(1, len(self.feature_names))
        }
        np.random.seed(seed)
        search = GridSearchCV(rf, param_dist, scoring="mean_squared_error")
        search.fit(X, y)
        report(search.grid_scores_, n_top=1)

    def bagging_rf_compare(self, n_estimator):
        case = 0
        for rseed in xrange(50):
            test_mse_1 = self.method_test(n_estimators=n_estimator, bagging=True, random_seed=rseed)
            test_mse_2 = self.method_test(n_estimators=n_estimator, bagging=False, random_seed=rseed)
            if test_mse_1 > test_mse_2:
                case += 1
            print '\n'
        print "Random forest is better than bagging in %d times out of 50 times" % case


'''
The test MSE of bagging isn't higher than random forest in every case.
Maybe in the case when p is very large it would have better performance, like figure 8-10 p=500
Can use a scikit-learn function to generate some data like this to test
'''
if __name__ == '__main__':
    dt_test = BaggingAndRandomForests()
    for i in xrange(10):
        dt_test.cv_judge(i)
