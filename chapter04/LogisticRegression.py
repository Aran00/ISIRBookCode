__author__ = 'Aran'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp


class LogisticRegression:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
        #self.df['Direction'] = self.df['Direction'].map(lambda x: 1 if x == 'Up' else 0)
        print self.df.columns
        self.y_col = 'Direction'
        self.x_cols = self.df.columns.tolist()
        self.x_cols.remove(self.y_col)
        print self.x_cols

    def show_correlation(self):
        cov_df = pd.DataFrame(np.corrcoef(self.df[self.x_cols], rowvar=0), columns=self.x_cols, index=self.x_cols)
        print "The correlation coefficients of each column is: \n", cov_df
        plt.scatter(range(self.df.shape[0]), self.df['Volume'], c='w')
        plt.show()

    def logistic_fit(self):
        '''
        The logit function would report error when y(Direction) is not transformed to 0/1
        So glm looks easier to use
        '''
        #model = smf.logit("Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume", data=self.df)
        model = smf.glm("Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume", data=self.df, family=sm.families.Binomial())
        result = model.fit()
        print result.summary()

if __name__ == '__main__':
    logis = LogisticRegression()
    #logis.show_correlation()
    logis.logistic_fit()