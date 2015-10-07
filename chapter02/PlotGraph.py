__author__ = 'ryu'

'''
The equal function in pandas as pairs() in R
http://stackoverflow.com/questions/2682144/matplotlib-analog-of-rs-pairs
'''

import pandas as pd
import matplotlib.pyplot as plt
from LoadData import LoadData

'''ISIR 2-3-5'''
class PlotGraph:
    def __init__(self):
        ld = LoadData()
        ld.load_csv()
        self.df = ld.df

    def plot_scatter(self):
        #plt.scatter(self.df['cylinders'], self.df['mpg'], c='w')
        self.df.plot(kind='scatter', x='cylinders', y='mpg', c='w')
        plt.show()

    def plot_box(self):
        '''It seems we have to transform the data first. Is there any better solution?'''
        '''
        dim = self.df.shape
        val_dict = {}
        for i in xrange(dim[0]):
            seri = self.df.ix[i]
            val_dict.setdefault(seri['cylinders'], [])
            val_dict[seri['cylinders']].append(seri['mpg'])
        keys = sorted(val_dict)
        data = []
        for key in keys:
            data.append(val_dict[key])
        plt.boxplot(data, labels=keys)
        '''
        '''Usen data frame function directly'''
        self.df.boxplot(column='mpg', by='cylinders')
        plt.show()

    def plot_hist(self):
        # plt.hist(self.df['mpg'], bins=9, range=(5, 50), color='w')
        self.df['mpg'].plot(kind='hist', bins=9, range=(5, 50), color='w')
        plt.show()

    '''Seems this function at most support 8 columns'''
    def plot_pairs(self):
        '''
        df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
        axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
        plt.tight_layout()
        plt.savefig('scatter_matrix.png')
        '''
        # print self.df.columns
        # df = pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )
        axes = pd.tools.plotting.scatter_matrix(self.df, alpha=0.2)
        # plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    pg = PlotGraph()
    # pg.plot_scatter()
    pg.plot_hist()
    # pg.plot_box()
    # pg.plot_pairs()