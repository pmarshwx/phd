import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def prepare_reliability(fcounts, ocounts, precision=0):
    """
    Filters forecast and observation counts.

    Parameters
    ----------
    fcounts : array_like
        The forecast counts at each probability threshold
    ocounts : array_like
        The observed counts at each probability threshold
    precision : integer (default 0)
        The number of decimal points in fcounts/ocounts

    Returns
    -------
    fcsts : array
        Forecast counts filtered so that true 0s and 100s are accounted for
    obs : array
        Observed counts filtered so that true 0s and 100s are accounted for
    fcst_percent : array
        The forecast probabilities
    obs_percent : array
        The observed probabilities

    """
    fcounts = np.asarray(fcounts)
    ocounts = np.asarray(ocounts)
    step = 100 / (fcounts.shape[0] - 2)
    fcst_percent = np.arange(step/2., 100+step, step)
    fcst_percent = fcst_percent / 10**precision
    fcsts = np.zeros_like(fcst_percent)
    obs = np.zeros_like(fcst_percent)
    fcsts[0] = fcounts[:2].sum()
    fcsts[1:] = fcounts[2:]
    obs[0] = ocounts[:2].sum()
    obs[1:] = ocounts[2:]
    obs_percent = obs / fcsts
    obs_percent[np.isnan(obs_percent)] = 0
    return fcsts, obs, fcst_percent, obs_percent


def reliability(ax=None, figsize=(10,10)):
    '''
    Draws the background of a reliability diagram.

    Parameters
    ----------
    ax : matplotlib axes instance (optional)
        The matplotlib axes instance on which to draw the performance diagram.
        If no axes is provided, one is created
    figsize : tuple (default (10,10))
        A tuple of the width, height in inches for the figure size, if not
        axes instance is provided

    Returns
    -------
    ax : matplotlib axes instance

    '''
    if not ax:
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1,1), direction="column",
                         axes_pad=1, add_all=True, label_mode="1",
                         share_all=True, cbar_location="bottom",
                         cbar_mode="none", cbar_size="5%", cbar_pad="5%")
        ax = grid[0]; cax = grid.cbar_axes[0]
    diag = np.linspace(0, 1)
    ax.plot(diag, diag, '--', lw=0.5, color='k')
    ax.set_ylabel('Observed Probability', size=14, labelpad=15)
    ax.set_xlabel('Forecast Probability', size=14, labelpad=15)
    ax.title.set_y(1.05)
    ax.set_title('Reliability Diagram', size=18,
                 verticalalignment='baseline')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 100+1e-6, 10) / 100.)
    ax.set_xticklabels(['%.1f' % (l) for l in ax.get_xticks()])
    ax.set_yticks(ax.get_xticks())
    ax.set_yticklabels(['%.1f' % (l) for l in ax.get_xticks()])
    minorLocator = MultipleLocator(0.05)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(axis='both', direction='out', which='major',
                    length=10, width=1, pad=10, top=False, right=False)
    ax.tick_params(axis='both', direction='out', which='minor',
                    length=6, width=0.75, top=False, right=False)
    ax.grid()
    return ax
