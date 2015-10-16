__author__ = 'ryu'

import numpy as np
from numpy import linalg as lg

data1 = [1, -2, 3, -4, 5]
arr1 = np.array(data1)
print arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
arr2 = np.array(data2)
print arr2
'''
print arr2.ndim
print len(arr2)     # equal to the last value
print arr2.shape    # (rowCount, colCount)

print "\n"
print np.zeros(10)
print np.zeros((2, 4))
print np.empty((2, 3))

print "\n"
arr1 = np.array([1, 2, 3], dtype=np.float64)
print arr1
arr3 = arr1.astype(np.int64)
print arr3
print arr1.dtype
arr2 = np.array([1, 2, 3], dtype=np.int32)
print arr2

print np.sqrt(arr2)
print np.square(arr2)
arr3 = np.where(arr2 > 4, 1, 0)
print arr3
'''

arr_slice = arr1[2:4]
arr_slice[0] = 4
arr1[3] = 6
print arr1

arr_slice1 = arr2[:, 0]
print arr_slice1
print arr2.T
arr3 = np.dot(arr2, arr2.T)
print arr3
print lg.inv(np.array([[1, 0], [0, 1]], dtype=np.int32))
print np.std(arr1)
print np.std(arr2)

np.linspace()