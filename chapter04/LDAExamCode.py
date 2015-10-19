__author__ = 'Aran'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas import DataFrame, Series


class LDAExamCode:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Smarket.csv", index_col=0)
        #self.df['Direction'] = self.df['Direction'].map(lambda x: 1 if x == 'Up' else 0)
        print self.df.columns
        self.y_col = 'Direction'
        self.x_cols = self.df.columns.tolist()
        self.x_cols.remove(self.y_col)
        print self.x_cols