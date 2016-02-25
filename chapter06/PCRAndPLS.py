__author__ = 'Aran'

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_decomposition import PLSRegression


''' Time to devise a global data loading function '''
class PCRAndPLS:
    def __init__(self):
        hitter = pd.read_csv("../../ISIRExerciseCode/dataset/Hitter.csv", index_col=0, na_values=['NA'])
        hitter = hitter.dropna()
        self.hitter = self.transform_label(hitter)
        self.y_col = 'Salary'
        self.y = self.hitter[self.y_col]
        self.x_cols = self.hitter.columns.tolist()
        self.x_cols.remove(self.y_col)
        self.X = self.hitter.ix[:, self.x_cols]

    def transform_label(self, hitter):
        #trans_cols = ["League", "Division", "NewLeague"]
        for col in hitter.columns.tolist():
            if isinstance(hitter.iloc[0][col], str):
                le = preprocessing.LabelEncoder()
                le.fit(np.unique(hitter[col]))
                hitter[col] = le.transform(hitter[col].values)
        return hitter

    def pcr_test(self):
        for i in xrange(1, 8):
            pca = decomposition.PCA(n_components=i)
            X_pca = pca.fit_transform(self.X)
            # print X_pca.shape, pca.explained_variance_ratio_
            clf = linear_model.LinearRegression()
            fit_result = clf.fit(X_pca, self.y)
            ''' Exact the same with the result of R predict function '''
            print fit_result.predict(X_pca[0:1, :])

    ''' Choose the best M for PCR '''
    def pcr_cv_test(self):
        pca = decomposition.PCA()
        linear = linear_model.LinearRegression()
        pipe = Pipeline(steps=[("pca", pca), ("linear", linear)])
        estimator = GridSearchCV(pipe, dict(pca__n_components=np.arange(1, 20)), cv=5)
        estimator.fit(self.X, self.y)
        print estimator.best_estimator_.named_steps  # ['pca'].n_components

    def pls_test(self):
        pls = PLSRegression()
        grid_cv = GridSearchCV(pls, dict(n_components=np.arange(1, 20)), cv=5)
        grid_cv.fit(self.X, self.y)
        print grid_cv.best_estimator_, grid_cv.best_params_['n_components']


if __name__ == '__main__':
    pap = PCRAndPLS()
    pap.pls_test()
