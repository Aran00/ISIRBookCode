__author__ = 'Aran'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from islrtools import tableplot as tp
from pandas import DataFrame, Series

class LogisticRegression:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
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

    '''
    The 2 list params: ndarray
    '''
    def output_binary_table(self, res, predict_probs, real_values, glm_fit=True):
        header = "predict   real"
        model = res.model
        print header
        '''
        #If transform y column to 0-1 in advance, the model.endog_names would be one variable, not a list
        output_data = DataFrame([[0, 0], [0, 0]], columns=[0,1], index=[0,1])
        zero_one_columns = [0, 1]
        '''
        zero_one_columns = self.get_real_zero_one_columns(res) if glm_fit else [0, 1]
        tp.output_table_with_prob(predict_probs, real_values, zero_one_col_texts=zero_one_columns)

    def get_real_zero_one_columns(self, res):
        model = res.model
        probs = res.fittedvalues
        # compare data1 and data2 to see what text is 1
        # Is it really necessary? Is the column in a fixed sequence that 1 is the former one?
        data1 = model.data.orig_endog.ix[probs.index[0], :]     # really up or down
        real_val = model.endog[probs.index[0]]
        real_col_index = 0 if data1[model.endog_names[0]] == 1 else 1
        real_val_one_column_index = real_col_index if real_val == model.endog.max else 1-real_col_index
        zero_one_columns = [model.endog_names[1 - real_val_one_column_index], model.endog_names[real_val_one_column_index]]
        return zero_one_columns

    def logistic_fit(self, glm_fit=True):
        '''
        The logit function would report error when y(Direction) is not transformed to 0/1
        So glm looks easier to use
        '''
        formula = "Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume"
        if glm_fit is True:
            model = smf.glm(formula, data=self.df, family=sm.families.Binomial())
        else:
            # In fact, this function has wrong fittedvalues, but it's predict value is still right.
            model = smf.logit(formula, data=self.df)
        result = model.fit()
        print result.summary()
        # In logit fit there are errors here. Not sure why...
        if glm_fit:
            self.output_binary_table(result, result.fittedvalues, model.endog.astype(int), glm_fit)

    def divide_train_set_and_fit(self, full_entities=True):
        train_data = self.df.ix[self.df['Year'] < 2005, :]
        test_data = self.df.ix[self.df.Year >= 2005, :]
        formula = "Direction~Lag1+Lag2"
        if full_entities is True:
            formula += "+Lag3+Lag4+Lag5+Volume"
        model = smf.glm(formula, data=train_data, family=sm.families.Binomial())
        result = model.fit()
        print result.summary()
        predict_result = result.predict(exog=test_data)
        real_val = test_data['Direction'].map(lambda x: 1 if x == 'Down' else 0)
        self.output_binary_table(result, predict_result, real_val)
        return result

    def predict_value(self, model_fit_result, exog):
        predict_result = model_fit_result.predict(exog=exog)
        print predict_result

    def test_logistic_coding(self):
        self.df['Direction'] = self.df['Direction'].map(lambda x: 1 if x == 'Up' else 0)
        self.logistic_fit(glm_fit=False)
        self.df['Direction'] = self.df['Direction'].map(lambda x: 1 - x)
        self.logistic_fit(glm_fit=False)


if __name__ == '__main__':
    logis = LogisticRegression()
    #logis.show_correlation()
    #logis.logistic_fit(glm_fit=True)
    result = logis.divide_train_set_and_fit(False)
    logis.predict_value(result, DataFrame([[1.2, 1.1], [1.5, -0.8]], columns=['Lag1', 'Lag2']))
