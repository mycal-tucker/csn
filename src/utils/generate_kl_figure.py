# AAAI won't allow me to use pgfplots, so I'll just create images here to plot.
# It's dumb, though.

import matplotlib.pyplot as plt
params = {'ytick.labelsize': 16}
plt.rcParams.update(params)

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-4, 4, N)
Y = np.linspace(-4, 4, N)
X, Y = np.meshgrid(X, Y)

p11 = np.array([2, 0])
p12 = np.array([-2, 0])
p21 = np.array([0, 2])
p22 = np.array([0, -2])

# mu0 = np.array([2, 2])
# mu1 = np.array([-2, -2])
mu0 = np.array([0.5, 0.5])
mu1 = np.array([-0.5, -0.5])

Sigma = np.array([[ 1. , 0], [0,  10]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


def multivariate_gaussian(pos, m, s):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = m.shape[0]
    Sigma_det = np.linalg.det(s)
    Sigma_inv = np.linalg.inv(s)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-m, Sigma_inv, pos-m)

    return np.exp(-fac / 2) / N


# The distribution on the variables X, Y packed into pos.
z1 = multivariate_gaussian(pos, mu0, Sigma)
z2 = multivariate_gaussian(pos, mu1, Sigma)

total_Z = (z1 + z2) / 2

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, total_Z, rstride=1, cstride=1, linewidth=10, antialiased=True,
                cmap='BuGn', alpha=0.5)

marker_size = 100
ax.scatter(p11[0], p11[1], np.array([0.01]), c='b', s=marker_size, marker='s')
ax.scatter(p12[0], p12[1], np.array([0.01]), c='b', s=marker_size, marker='s')
ax.scatter(p21[0], p21[1], np.array([0.01]), c='r', s=marker_size, marker='o')
ax.scatter(p22[0], p22[1], np.array([0.01]), c='r', s=marker_size, marker='o')

# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(0, 0.12)
ax.set_zticks(np.linspace(0, 0.10, 3))
ax.view_init(27, -70)

plt.show()
