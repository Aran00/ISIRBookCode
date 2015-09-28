__author__ = 'ryu'

import numpy as np

A = np.arange(1, 17).reshape(4, 4).T
B = A[[[0], [2]], [1, 3]]
print B
'''
Another method from stackoverflow:
http://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array
'''
row_idx = np.array([0, 2])
print A[row_idx[:, None], [1, 3]]

C = A[0:3, 1:4]
#print C
D = A[0:2, :]
print D
E = A[:, 0:2]
print E
F = A[-3:-1]
print F
# except part has no correspondence
print A.shape

