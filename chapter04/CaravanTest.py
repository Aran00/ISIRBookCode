__author__ = 'ryu'

import pandas as pd
import tableplot as tp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.preprocessing as pp
from sklearn import neighbors


class CaravanTest:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Caravan.csv", index_col=0)
        self.train_set = self.df.ix[1000:, :]
        self.test_set = self.df.ix[0:1000, :]
        self.y_col = 'Purchase'
        all_cols = self.df.columns.values
        self.x_cols = all_cols[all_cols != self.y_col]
        print self.x_cols.shape

    def fit_with_knn(self, n_neighbors):
        train_X = pp.scale(self.train_set[self.x_cols].values.astype(float))
        train_y = self.train_set[self.y_col]
        weights = 'uniform'
        #weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(train_X, train_y)
        test_X = pp.scale(self.test_set[self.x_cols].values.astype(float))
        test_y = self.test_set[self.y_col].values
        preds = clf.predict(test_X)
        tp.output_table(preds, test_y)

    def fit_with_logistic(self, threshold=0.5):
        formula = "%s~%s" % (self.y_col, "+".join(self.x_cols))
        model = smf.glm(formula, data=self.train_set, family=sm.families.Binomial())
        result = model.fit()
        predict_probs = result.predict(exog=self.test_set)
        real_values = self.test_set[self.y_col].map(lambda x: 1 if x == 'No' else 0)
        tp.output_table_with_prob(predict_probs, real_values, threshold=threshold, zero_one_col_texts=["Yes", "No"])


if __name__ == '__main__':
    example = CaravanTest()
    example.fit_with_logistic(0.75)
    #example.fit_with_knn(1)