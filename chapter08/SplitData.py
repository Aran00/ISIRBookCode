__author__ = 'Aran'

import numpy as np
from numpy import random
from operator import itemgetter


def split_data(data, train_count, rand_seed=1):
    np.random.seed(rand_seed)
    train_index = random.choice(data.index, train_count, replace=False)
    # print train_index
    train_cond = data.index.isin(train_index)
    return data.ix[train_cond], data.ix[~train_cond]

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
