import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import unravel_index
# from pylab import cm, colorbar

# https://stackoverflow.com/questions/36700196/plot-contours-of-distribution-on-all-three-axes-in-3d-plot
# do kernel density estimation to get smooth estimate of distribution
# make grid of data


def to_int(value, state):
    if state == 'max':
        return np.ceil(value)
    elif state == 'min':
        return np.floor(value)


def plot3d(prop_name, data, figpath, figsize=10, s=1, alpha=0.75,
           cmap='RdBu_r', levels=20, ax_limits=None, ngrids=20j):

    """
    plot a scattering 3D figure with distributions on
    three planes
    ----------
    prop_name: list
        a list of the three variables
    data: pandas.DataFrame
        a DataFrame recording the points
    figpath: string
        the figure path
    figsize: float, 10
        figure size
    s: float, 1
        the size of each point
    alpha: float, 0.75
        the transparency
    cmap: string, 'RdBu_r'
        the color
    levels: int, 20
        the number of the contour lines
    ax_limits: dict, None
        the limits of the axes in three directions
    ngrids: int, 20
        number of the grids each sides
    """

    assert len(prop_name) == 3

    if ax_limits == None:
        ax_limits = {
            prop_name[0]: [np.floor(data[prop_name[0]].min()),
                           np.ceil(data[prop_name[0]].max())],
            prop_name[1]: [np.floor(data[prop_name[1]].min()),
                           np.ceil(data[prop_name[1]].max())],
            prop_name[2]: [np.floor(data[prop_name[2]].min()),
                           np.ceil(data[prop_name[2]].max())]
        }

    xmin = ax_limits[prop_name[0]][0]
    xmax = ax_limits[prop_name[0]][1]
    ymin = ax_limits[prop_name[1]][0]
    ymax = ax_limits[prop_name[1]][1]
    zmin = ax_limits[prop_name[2]][0]
    zmax = ax_limits[prop_name[2]][1]

    x, y, z = np.mgrid[xmin:xmax:ngrids,
                       ymin:ymax:ngrids,
                       zmin:zmax:ngrids]


    # Convert DataFrame to Numpy array
    data = data.to_numpy().T

    # Compute kernel density
    kernel = sp.stats.gaussian_kde(data)
    positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    density = np.reshape(kernel(positions).T, x.shape)

    d1, d2, d3 = unravel_index(density.argmax(), density.shape)
    highest_freq_values = [x[d1,d2,d3], y[d1,d2,d3], z[d1,d2,d3]]

    # plot data
    ax = plt.subplot(projection='3d')

    # Set figure size
    fig = plt.gcf()
    fig.set_size_inches(figsize, figsize, figsize)

    ax.scatter(data[0, :], data[1, :], data[2, :], s=s, marker='o', c='k')

    ax.set_xlabel(prop_name[0], fontsize=22)
    ax.set_ylabel(prop_name[1], fontsize=22)
    ax.set_zlabel(prop_name[2], fontsize=22)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)    

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_zlim((zmin, zmax))
    
    print('plot projection of density onto x-axis')
    plotdat = np.sum(density, axis=0)  # summing up density along z-axis
    plotdat = plotdat / np.max(plotdat)
    ploty, plotz = np.mgrid[ymin:ymax:ngrids, zmin:zmax:ngrids]
    
    colorx = ax.contourf(plotdat, ploty, plotz, levels=levels,
                        alpha=alpha, cmap=cmap, offset=xmin, zdir='x')

    print('plot projection of density onto y-axis')
    plotdat = np.sum(density, axis=1)  # summing up density along y-axis
    plotdat = plotdat / np.max(plotdat)
    plotx, plotz = np.mgrid[xmin:xmax:ngrids, zmin:zmax:ngrids]
    colory = ax.contourf(plotx, plotdat, plotz, levels=levels,
                        alpha=alpha, cmap=cmap, offset=ymax, zdir='y')

    print('plot projection of density onto z-axis')
    plotdat = np.sum(density, axis=2)
    plotdat = plotdat / np.max(plotdat)
    plotx, ploty = np.mgrid[xmin:xmax:ngrids, ymin:ymax:ngrids]
    colorz = ax.contourf(plotx, ploty, plotdat, levels=levels,
                        alpha=alpha, cmap=cmap, offset=zmin, zdir='z')

    cbar = fig.colorbar(colorx, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(figpath)
    
    return highest_freq_values


def plot_3props_distribution(props_name, n_samplings, file_path, fig_path, result_path):
    df = pd.read_csv(file_path)
    df = df.sample(n=n_samplings)
    df = df[props_name]
    print(df)

    highest_freq_values = plot3d(props_name, df, fig_path, figsize=20)
    with open(result_path, "w") as ptr:
        header = "\t".join(props_name)
        values = "\t".join([str(v) for v in highest_freq_values])
        ptr.write(header+"\n")
        ptr.write(values+"\n")


if __name__ == '__main__':
    """
    arguments
    """
    prop_name = ['logP', 'tPSA', 'QED']
    n_samplings = 100000
    file_path = '/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/prop_temp.csv'
    fig_path = './var3_scattering.png'
    result_path = './highest_freq_values.txt'
    
    plot_3props_distribution(prop_name, n_samplings, file_path, fig_path, result_path)
