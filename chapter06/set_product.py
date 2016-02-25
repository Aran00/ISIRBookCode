__author__ = 'Aran'

import itertools as it


a = {'a', 'b'}
b = {'c', 'd'}
a.add('a')
print a
a.add('d')
print a

for ab in it.product(a, b):
    print ab
