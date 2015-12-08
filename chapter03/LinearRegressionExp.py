from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, Series
from statsmodels.stats.api import anova_lm
from islrtools import poly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy, scipy.stats


class LinearRegressionExp:
    def __init__(self):
        self.df = None

    def read_boston_data(self):
        self.df = pd.read_csv("~/Documents/Boston.csv")

    def plot_scatter(self):
        self.df.plot(kind='scatter', x='lstat', y='medv', c='w')
        plt.show()

    def stat_single_regression(self):
        y = self.df['medv']
        X = self.df[['lstat']]
        X = sm.add_constant(X)      # Add constant for linear regression
        results = sm.OLS(y, X).fit()

        print results.summary()
        print "The coeffcients are: \n", results.params
        print "The coeffcients intervals are: \n", results.conf_int()
        print "The predicted values are: ", results.predict(sm.add_constant([1, 10, 15]))

        graph_x = np.linspace(min(self.df['lstat']), max(self.df['lstat']))
        graph_y = results.predict(sm.add_constant(graph_x))
        self.df.plot(kind='scatter', x='lstat', y='medv', c="w")
        plt.plot(graph_x, graph_y)
        plt.show()

    def stat_multi_regression(self):
        y = self.df['medv']
        X = self.df[['lstat', 'age']]
        X = sm.add_constant(X)      # Add constant for linear regression
        results = sm.OLS(y, X).fit()
        print results.summary()

    def stat_r_style_regression(self):
        ''' See the detailed docs in http://statsmodels.sourceforge.net/devel/example_formulas.html#ols-regression-using-formulas'''
        mod = smf.ols(formula="medv ~ lstat + age", data=self.df)
        res = mod.fit()
        print res.summary()

    def stat_multi_regression_b(self):
        ''' all columns without '''
        y = self.df['medv']
        columns = self.df.columns.values.tolist()
        columns.remove('medv')
        #columns.remove('age')
        X = self.df[columns]
        X = sm.add_constant(X)      # Add constant for linear regression
        results = sm.OLS(y, X).fit()
        self.fit = results
        '''
        print results.summary()
        print "R^2 is ", results.rsquared
        print "RSE is ", np.sqrt(results.mse_resid)
        print "RSS is ", results.ssr
        '''

    def stat_multi_include_interaction(self):
        mod = smf.ols(formula="medv ~ lstat * age", data=self.df)
        res = mod.fit()
        print res.summary()

    def stat_multi_include_poly(self):
        y = self.df['medv']
        x_vec = self.df['lstat']
        Z, norm2, alpha = poly.ortho_poly_fit(x_vec.tolist(), 5)
        results = sm.OLS(y, Z).fit()

        print results.summary()
        print "The coeffcients are: \n", results.params
        print "The coeffcients intervals are: \n", results.conf_int()
        print "The predicted values are: ", results.predict(poly.ortho_poly_predict([1, 10, 15], alpha, norm2, 5))

    def stat_multi_include_calculation(self):
        #mod = smf.ols(formula="medv ~ lstat + I(lstat ** 2)", data=self.df)
        mod = smf.ols(formula="medv ~ np.log(rm)", data=self.df)
        res = mod.fit()
        self.fit2 = res
        print res.summary()

    def anova_test(self):
        self.stat_multi_regression_b()
        self.stat_multi_include_calculation()
        table1 = anova_lm(self.fit, self.fit2)
        print table1

    def try_quality_variables(self):
        car_df = pd.read_csv("~/Documents/Carseats.csv", index_col=0)
        car_df['Income:Advertising'] = car_df['Income']*car_df['Advertising']
        car_df['Price:Age'] = car_df['Price']*car_df['Age']
        car_df['ShelveLocGood'] = car_df['ShelveLoc'].map(lambda x: 1 if x == "Good" else 0)
        car_df['ShelveLocMedium'] = car_df['ShelveLoc'].map(lambda x: 1 if x == "Medium" else 0)
        yes_no_func = lambda x: 1 if x == "Yes" else 0
        car_df['UrbanYes'] = car_df['Urban'].map(yes_no_func)
        car_df['USYes'] = car_df['US'].map(yes_no_func)

        columns = car_df.columns.values.tolist()
        columns.remove('Sales')
        columns.remove('ShelveLoc')
        columns.remove('Urban')
        columns.remove('US')
        #columns.remove('age')
        X = car_df[columns]
        X = sm.add_constant(X)
        y = car_df['Sales']
        results = sm.OLS(y, X).fit()
        print results.summary()

if __name__ == '__main__':
    lr = LinearRegressionExp()
    lr.read_boston_data()
    #lr.plot_scatter()
    #lr.stat_regression()
    #lr.stat_multi_regression_b()
    #lr.stat_r_style_regression()
    #lr.stat_multi_include_calculation()
    #lr.anova_test()
    #lr.try_quality_variables()
    lr.stat_multi_include_poly()

'''
clf = linear_model.LinearRegression()
lr = clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print lr
'''
