__author__ = 'Aran'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

# x and y are 1-dim ndarrays
def outer(dim_1_x, dim_1_y, func=np.multiply):
    dim_2_x, dim_2_y = np.meshgrid(dim_1_x, dim_1_y)
    return func(dim_2_x, dim_2_y), dim_2_x, dim_2_y

def test_func(dim_2_x, dim_2_y):
    return np.cos(dim_2_y)/(1 + dim_2_x**2)

x = np.linspace(-np.pi, np.pi, num=50)
y = x
f, X, Y = outer(x, y, test_func)
res = (f - f.T)/2
fig = plt.figure()
'''
ax1 = fig.add_subplot(111)
CS = ax1.contour(X, Y, res, levels=np.arange(-1, 1, 0.1))
ax1.clabel(CS, inline=1, fontsize=10)
'''
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, res, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("func(x,y)")
plt.show()