import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
mpl.use('Agg')


def hist_box(data_frame, field, name="hist_box", path="./", title=None):

    title = title if title else field
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    data_frame[field].plot.hist(bins=100, title=title, ax=axs[0])
    data_frame.boxplot(field, ax=axs[1])
    plt.title(title)
    plt.suptitle("")

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()


def hist(data_frame, field, name="hist", path="./", title=None):
    title = title if title else field

    plt.hist(data_frame[field])
    plt.title(title)
    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()


def hist_box_list(data_list, name="hist_box", path="./", title=None):
    fig, axs = plt.subplots(1,2,figsize=(10,4))

    axs[0].hist(data_list, bins=100)
    axs[0].set_title(title)
    axs[1].boxplot(data_list)
    axs[1].set_title(title)

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()


def kde_plot(df, save_path, xlabel, ylabel, xlimit=None,
             figsize=(6.5, 5), lengend=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    df.plot.kde(ax=ax, legend=lengend, xlim=xlimit)
    if lengend:
        ax.legend(fontsize=14)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig(save_path, bbox_inches="tight")