import os, sys
import datetime
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator


def performance_diagram(**kwargs):
    """
    Draws the background of a performance diagram.

    Parameters
    ----------
    ax : matplotlib axes instance (optional)
        The matplotlib axes instance on which to draw the performance diagram.
        If no axes is provided, one is created
    figsize : tuple (default (10,10))
        A tuple of the width, height in inches for the figure size, if not
        axes instance is provided
    bias_lines : array_like
        A sequence of values corresponding to the lines of bias to plot
    csi_lines : array_like
        A sequence of values corresponding to the lines of csi to plot

    Returns
    -------
    ax : matplotlib axes instance

    """
    ax = kwargs.get('ax', None)
    figsize = kwargs.get('figsize', (10,10))
    bias_lines = kwargs.get('bias_lines', None)
    csi_lines = kwargs.get('csi_lines', None)

    if not ax:
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1,1), direction="column",
                         axes_pad=1, add_all=True, label_mode="1",
                         share_all=True, cbar_location="bottom",
                         cbar_mode="none", cbar_size="5%", cbar_pad="5%")
        ax = grid[0]; cax = grid.cbar_axes[0]

    if not bias_lines:
        bias_lines = np.array([0.1, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 2, 3, 5, 10])

    if not csi_lines:
        csi_lines = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                              0.6, 0.7, 0.8, 0.9])
    sr_lines = np.linspace(0, 1, 100)
    pod_lines = np.linspace(0, 1, 500)
    # Plot Bias Lines
    for x in bias_lines:
        xx = pod_lines/x
        ax.plot(xx, pod_lines, linestyle='--', color='k', linewidth=1)
        b_offset = 0.03
        if x > 1:
            xxx = xx[-1]
            yyy = pod_lines[-1] + b_offset
        elif x < 1:
            min_xx = np.abs(xx - 1)
            min_ind = np.where(min_xx == min_xx.min())
            xxx = 1.0025 + b_offset
            yyy = pod_lines[min_ind]
            yyy = sum(yyy) / len(yyy)
        else:
            xxx = xx[-1] + b_offset
            yyy = pod_lines[-1] + b_offset
        ax.text(xxx, yyy, x, horizontalalignment='center',
                verticalalignment='center', bbox=dict(fc=(1,1,1,1),
                ec=(1,1,1,1), boxstyle="round, pad=0.33, rounding_size=0.5",
                lw=1))
    # Plot CSI Lines
    for x in csi_lines:
        xx = (1/x) - (1/pod_lines) + 1
        xx = np.ma.asanyarray(xx)
        xx[xx<0] = np.ma.masked
        xx = 1/xx
        ax.plot(xx, pod_lines, linestyle='-', color='k', linewidth=1)
        xxx = 0.95
        min_xx = np.abs(xx - xxx)
        min_ind = np.where(min_xx == min_xx.min())
        yyy = pod_lines[min_ind]
        ax.text(xxx, yyy, x, horizontalalignment='center',
                verticalalignment='center', bbox=dict(fc=(1,1,1,1),
                ec=(1,1,1,1), boxstyle="round, pad=0.33, rounding_size=0.5",
                lw=1))
    ax.set_xlabel('\nSuccess Ratio (1-FAR)', size=14)
    ax.set_xticks(np.arange(0, 1+1e-6, 0.1))
    ax.set_xlim(0, 1)
    ax.set_ylabel('Probability of Detection (POD)\n', size=14)
    ax.set_yticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_xticks())
    ax.set_ylim(0, 1)
    ax.set_title('Performance Diagram\n\n', size=18,
                 verticalalignment='baseline')
    minorLocator = MultipleLocator(0.05)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(axis='both', direction='out', which='major',
                    length=10, width=1, pad=10, top=False, right=False)
    ax.tick_params(axis='both', direction='out', which='minor',
                    length=6, width=0.75, top=False, right=False)
    return ax