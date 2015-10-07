__author__ = 'ryu'

import csv
import pandas as pd
# from pandas import Series, DataFrame

'''ISIR 2-3-4'''
class LoadData:
    def __init__(self):
        self.df = None

    def load_csv(self):
        df = pd.read_table('files/Auto.data', sep='\s+', na_values=['?'])
        self.df = df

    def load_csv_without_pandas(self):
        with open('files/Auto.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                print row

if __name__ == '__main__':
    ld = LoadData()
    ld.load_csv()
    print 'df.shape=', ld.df.shape
    print ld.df.ix[32]