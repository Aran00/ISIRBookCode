__author__ = 'ryu'

import random
import numpy as np


class SampleTest:
    def __init__(self):
        pass

    @staticmethod
    def random_test():
        random.seed(1)
        print random.sample(xrange(4), 4)

    @staticmethod
    def numpy_test():
        np.random.seed(1)
        print np.random.choice(4, 4, True, [0.5, 0.2, 0.2, 0.1])


if __name__ == '__main__':
    exp = SampleTest()
    exp.numpy_test()