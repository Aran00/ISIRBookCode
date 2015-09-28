__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


'''
fig = plt.figure()
ax = fig.add_subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=2)
ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
ax.set_ylim(-2, 2)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
r = np.arange(0, 1, 0.001)
theta = 2*2*np.pi*r
line, = ax.plot(theta, r, color="#ee8d18", lw=3)
ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'o')
ax.annotate('a polar annotation',
    xy=(thistheta, thisr), # theta, radius
    xytext=(0.05, 0.05), # fraction, fraction
    textcoords='figure fraction',
    arrowprops=dict(facecolor='black', shrink=0.05),
    horizontalalignment='left',
    verticalalignment='bottom',
)
plt.show()


img = mpimg.imread('files/stinkbug.png')
#print img
imgplot = plt.imshow(img)
imgplot.set_clim(0.0, 0.7)
lum_img = img[:, :, 0]
print lum_img
#plt.hist(lum_img.flatten(), 256, range=(0.0, 1.0), fc='k', ec='k')
#imgplot = plt.imshow(lum_img)
#imgplot.set_cmap('coolwarm')
#plt.colorbar()
plt.show()
'''

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()
